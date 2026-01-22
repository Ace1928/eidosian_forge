from urllib3.exceptions import HTTPError as BaseHTTPError
from .compat import JSONDecodeError as CompatJSONDecodeError
class ChunkedEncodingError(RequestException):
    """The server declared chunked encoding but sent an invalid chunk."""