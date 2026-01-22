import warnings
class NullOutputStream:
    """Acts like a file, but discard all output."""

    def __init__(self, encoding):
        self.encoding = encoding

    def write(self, data):
        pass

    def writelines(self, data):
        pass

    def close(self):
        pass