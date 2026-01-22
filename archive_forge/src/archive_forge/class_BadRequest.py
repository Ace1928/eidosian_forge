import inspect
import sys
from magnumclient.i18n import _
class BadRequest(HTTPClientError):
    """HTTP 400 - Bad Request.

    The request cannot be fulfilled due to bad syntax.
    """
    http_status = 400
    message = _('Bad Request')