import requests
class StreamParseError(RuntimeError):

    def __init__(self, reason):
        self.msg = reason