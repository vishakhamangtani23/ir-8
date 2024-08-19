import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def read_text_files(file_paths):
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())
    return documents

def compute_tfidf(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

def create_dataframe(tfidf_matrix, feature_names, file_names):
    df = pd.DataFrame(tfidf_matrix.toarray(), index=file_names, columns=feature_names)
    return df

def main(directory_path, file_names):
    # Read text files
    file_paths = [os.path.join(directory_path, file_name) for file_name in file_names]
    documents = read_text_files(file_paths)

    # Compute TF-IDF
    tfidf_matrix, feature_names = compute_tfidf(documents)

    # Create DataFrame with documents as rows and words as columns
    df = create_dataframe(tfidf_matrix, feature_names, file_names)

    # Print DataFrame
    print(df)

# Directory containing text files and their names
directory_path = 'C:\\Users\\Exam\\PycharmProjects\\vishakha\\venv\\Documents'
file_names = [f'Document{i}.txt' for i in range(1, 11)]  # Assuming file names are Document1.txt, ..., Document10.txt

main(directory_path, file_names)
