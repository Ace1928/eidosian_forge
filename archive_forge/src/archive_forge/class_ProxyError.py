from urllib3.exceptions import HTTPError as BaseHTTPError
from .compat import JSONDecodeError as CompatJSONDecodeError
class ProxyError(ConnectionError):
    """A proxy error occurred."""