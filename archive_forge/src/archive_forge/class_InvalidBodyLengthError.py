import h2.errors
class InvalidBodyLengthError(ProtocolError):
    """
    The remote peer sent more or less data that the Content-Length header
    indicated.

    .. versionadded:: 2.0.0
    """

    def __init__(self, expected, actual):
        self.expected_length = expected
        self.actual_length = actual

    def __str__(self):
        return 'InvalidBodyLengthError: Expected %d bytes, received %d' % (self.expected_length, self.actual_length)