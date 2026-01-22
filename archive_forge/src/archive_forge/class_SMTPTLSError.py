from typing import Optional
class SMTPTLSError(ESMTPClientError):
    """
    An error occurred while negiotiating for transport security.

    This is considered a non-fatal error (the connection will not be dropped).
    """