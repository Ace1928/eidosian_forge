from typing import Optional
class AUTHRequiredError(ESMTPClientError):
    """
    Authentication was required but the server does not support it.

    This is considered a non-fatal error (the connection will not be dropped).
    """