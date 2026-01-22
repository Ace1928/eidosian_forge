class ImportError(Exception):
    """The base exception class for all import processing exceptions."""

    def __init__(self):
        super(ImportError, self).__init__(self._fmt % self.__dict__)