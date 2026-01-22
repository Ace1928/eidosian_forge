from typing import Optional
class SMTPTimeoutError(SMTPClientError):
    """
    Failed to receive a response from the server in the expected time period.

    This is considered a fatal error.  A retry will be made.
    """

    def __init__(self, code, resp, log=None, addresses=None, isFatal=True, retry=True):
        SMTPClientError.__init__(self, code, resp, log, addresses, isFatal, retry)