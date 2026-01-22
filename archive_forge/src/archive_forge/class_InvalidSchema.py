from urllib3.exceptions import HTTPError as BaseHTTPError
from .compat import JSONDecodeError as CompatJSONDecodeError
class InvalidSchema(RequestException, ValueError):
    """The URL scheme provided is either invalid or unsupported."""