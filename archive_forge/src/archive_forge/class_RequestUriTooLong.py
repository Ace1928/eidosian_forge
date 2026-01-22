import inspect
import sys
from magnumclient.i18n import _
class RequestUriTooLong(HTTPClientError):
    """HTTP 414 - Request-URI Too Long.

    The URI provided was too long for the server to process.
    """
    http_status = 414
    message = _('Request-URI Too Long')