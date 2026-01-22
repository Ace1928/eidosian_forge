import inspect
import sys
from magnumclient.i18n import _
class RequestTimeout(HTTPClientError):
    """HTTP 408 - Request Timeout.

    The server timed out waiting for the request.
    """
    http_status = 408
    message = _('Request Timeout')