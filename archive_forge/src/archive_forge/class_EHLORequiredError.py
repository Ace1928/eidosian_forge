from typing import Optional
class EHLORequiredError(ESMTPClientError):
    """
    The server does not support EHLO.

    This is considered a non-fatal error (the connection will not be dropped).
    """