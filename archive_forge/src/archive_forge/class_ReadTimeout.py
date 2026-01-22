from urllib3.exceptions import HTTPError as BaseHTTPError
from .compat import JSONDecodeError as CompatJSONDecodeError
class ReadTimeout(Timeout):
    """The server did not send any data in the allotted amount of time."""