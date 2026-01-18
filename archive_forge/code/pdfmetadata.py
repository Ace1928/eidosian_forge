import os
from PyPDF2 import PdfFileReader

# Directory containing the PDF files
pdf_directory = "path/to/pdf/directory"

# Iterate over the PDF files in the directory
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        # Open the PDF file
        pdf_path = os.path.join(pdf_directory, filename)
        with open(pdf_path, "rb") as file:
            # Create a PDF reader object
            pdf_reader = PdfFileReader(file)

            # Extract metadata
            metadata = pdf_reader.getDocumentInfo()
            author = metadata.get("/Author", "Unknown")
            title = metadata.get("/Title", "Unknown")

            print(f"File: {filename}")
            print(f"Author: {author}")
            print(f"Title: {title}")
            print("---")

print("Metadata extraction completed.")
