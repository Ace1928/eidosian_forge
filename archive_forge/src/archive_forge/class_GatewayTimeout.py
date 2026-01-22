import inspect
import sys
from magnumclient.i18n import _
class GatewayTimeout(HttpServerError):
    """HTTP 504 - Gateway Timeout.

    The server was acting as a gateway or proxy and did not receive a timely
    response from the upstream server.
    """
    http_status = 504
    message = _('Gateway Timeout')