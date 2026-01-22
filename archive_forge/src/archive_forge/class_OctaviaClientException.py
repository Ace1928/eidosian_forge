class OctaviaClientException(Exception):
    """The base exception class for all exceptions this library raises."""

    def __init__(self, code, message=None, request_id=None):
        self.code = code
        self.message = message or self.__class__.message
        super().__init__(self.message)
        self.request_id = request_id

    def __str__(self):
        return '%s (HTTP %s) (Request-ID: %s)' % (self.message, self.code, self.request_id)