import inspect
import sys
from magnumclient.i18n import _
class HTTPClientError(HttpError):
    """Client-side HTTP error.

    Exception for cases in which the client seems to have erred.
    """
    message = _('HTTP Client Error')