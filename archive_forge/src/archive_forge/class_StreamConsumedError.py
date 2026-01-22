from urllib3.exceptions import HTTPError as BaseHTTPError
from .compat import JSONDecodeError as CompatJSONDecodeError
class StreamConsumedError(RequestException, TypeError):
    """The content for this response was already consumed."""