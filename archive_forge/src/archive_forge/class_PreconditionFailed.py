import inspect
import sys
from magnumclient.i18n import _
class PreconditionFailed(HTTPClientError):
    """HTTP 412 - Precondition Failed.

    The server does not meet one of the preconditions that the requester
    put on the request.
    """
    http_status = 412
    message = _('Precondition Failed')