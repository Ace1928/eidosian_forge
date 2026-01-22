from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
class ReconfigurationNotSupported(BzrDirError):
    _fmt = "Requested reconfiguration of '%(display_url)s' is not supported."