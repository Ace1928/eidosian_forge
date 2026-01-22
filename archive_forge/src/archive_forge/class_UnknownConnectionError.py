from keystoneauth1.exceptions import base
class UnknownConnectionError(ConnectionError):
    """An error was encountered but we don't know what it is."""

    def __init__(self, msg, original):
        super(UnknownConnectionError, self).__init__(msg)
        self.original = original