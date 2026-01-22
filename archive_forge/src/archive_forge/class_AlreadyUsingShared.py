from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
class AlreadyUsingShared(BzrDirError):
    _fmt = "'%(display_url)s' is already using a shared repository."