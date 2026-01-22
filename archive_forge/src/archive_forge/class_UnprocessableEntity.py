import inspect
import sys
from magnumclient.i18n import _
class UnprocessableEntity(HTTPClientError):
    """HTTP 422 - Unprocessable Entity.

    The request was well-formed but was unable to be followed due to semantic
    errors.
    """
    http_status = 422
    message = _('Unprocessable Entity')