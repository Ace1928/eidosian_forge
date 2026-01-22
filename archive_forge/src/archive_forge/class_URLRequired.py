from urllib3.exceptions import HTTPError as BaseHTTPError
from .compat import JSONDecodeError as CompatJSONDecodeError
class URLRequired(RequestException):
    """A valid URL is required to make a request."""