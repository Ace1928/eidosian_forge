import inspect
import sys
from magnumclient.i18n import _
class HTTPRedirection(HttpError):
    """HTTP Redirection."""
    message = _('HTTP Redirection')