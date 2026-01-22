import inspect
import sys
from magnumclient.i18n import _
class Forbidden(HTTPClientError):
    """HTTP 403 - Forbidden.

    The request was a valid request, but the server is refusing to respond
    to it.
    """
    http_status = 403
    message = _('Forbidden')