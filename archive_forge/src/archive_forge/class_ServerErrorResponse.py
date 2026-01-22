from typing import Optional
class ServerErrorResponse(POP3ClientError):
    """
    An error indicating that the server returned an error response to a
    request.

    @ivar consumer: See L{__init__}
    """

    def __init__(self, reason, consumer=None):
        """
        @type reason: L{bytes}
        @param reason: The server response minus the status indicator.

        @type consumer: callable that takes L{object}
        @param consumer: The function meant to handle the values for a
            multi-line response.
        """
        POP3ClientError.__init__(self, reason)
        self.consumer = consumer