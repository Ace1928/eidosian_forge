from typing import Optional
class AUTHDeclinedError(ESMTPClientError):
    """
    The server rejected our credentials.

    Either the username, password, or challenge response
    given to the server was rejected.

    This is considered a non-fatal error (the connection will not be
    dropped).
    """