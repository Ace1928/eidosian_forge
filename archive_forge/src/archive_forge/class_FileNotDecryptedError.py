class FileNotDecryptedError(PdfReadError):
    """
    Raised when a PDF file that has been encrypted
    (meaning it requires a password to be accessed) has not been successfully
    decrypted.
    """