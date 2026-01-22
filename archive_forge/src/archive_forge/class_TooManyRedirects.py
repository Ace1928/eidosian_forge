from urllib3.exceptions import HTTPError as BaseHTTPError
from .compat import JSONDecodeError as CompatJSONDecodeError
class TooManyRedirects(RequestException):
    """Too many redirects."""