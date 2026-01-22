import inspect
import sys
from magnumclient.i18n import _
class UnsupportedMediaType(HTTPClientError):
    """HTTP 415 - Unsupported Media Type.

    The request entity has a media type which the server or resource does
    not support.
    """
    http_status = 415
    message = _('Unsupported Media Type')