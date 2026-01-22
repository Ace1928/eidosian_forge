from PyPDF2 import PdfMerger

# Create a PdfMerger object
merger = PdfMerger()

# List of PDF files to merge
pdf_files = ["file1.pdf", "file2.pdf", "file3.pdf"]

# Append each PDF file to the merger object
for pdf_file in pdf_files:
    merger.append(pdf_file)

# Write the merged PDF to a new file
merger.write("merged.pdf")
merger.close()
