import io
import urllib.response
class ContentTooShortError(URLError):
    """Exception raised when downloaded size does not match content-length."""

    def __init__(self, message, content):
        URLError.__init__(self, message)
        self.content = content