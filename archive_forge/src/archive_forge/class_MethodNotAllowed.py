import inspect
import sys
from magnumclient.i18n import _
class MethodNotAllowed(HTTPClientError):
    """HTTP 405 - Method Not Allowed.

    A request was made of a resource using a request method not supported
    by that resource.
    """
    http_status = 405
    message = _('Method Not Allowed')