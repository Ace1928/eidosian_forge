from typing import Optional
class TLSNotSupportedError(POP3ClientError):
    """
    An error indicating secure authentication was required but the server does
    not support TLS.
    """