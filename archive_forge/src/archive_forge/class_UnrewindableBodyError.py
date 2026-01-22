from urllib3.exceptions import HTTPError as BaseHTTPError
from .compat import JSONDecodeError as CompatJSONDecodeError
class UnrewindableBodyError(RequestException):
    """Requests encountered an error when trying to rewind a body."""