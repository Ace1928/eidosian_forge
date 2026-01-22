from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
class AlreadyWithNoTrees(BzrDirError):
    _fmt = "Shared repository '%(display_url)s' already doesn't create working trees."