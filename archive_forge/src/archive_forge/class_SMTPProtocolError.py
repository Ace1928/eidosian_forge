from typing import Optional
class SMTPProtocolError(SMTPClientError):
    """
    The server sent a mangled response.

    This is considered a fatal error.  A retry will not be made.
    """

    def __init__(self, code, resp, log=None, addresses=None, isFatal=True, retry=False):
        SMTPClientError.__init__(self, code, resp, log, addresses, isFatal, retry)