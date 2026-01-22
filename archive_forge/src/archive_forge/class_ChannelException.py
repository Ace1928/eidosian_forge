import socket
class ChannelException(SSHException):
    """
    Exception raised when an attempt to open a new `.Channel` fails.

    :param int code: the error code returned by the server

    .. versionadded:: 1.6
    """

    def __init__(self, code, text):
        SSHException.__init__(self, code, text)
        self.code = code
        self.text = text

    def __str__(self):
        return 'ChannelException({!r}, {!r})'.format(self.code, self.text)