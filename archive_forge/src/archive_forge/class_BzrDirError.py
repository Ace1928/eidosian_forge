from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
class BzrDirError(errors.BzrError):

    def __init__(self, controldir):
        display_url = urlutils.unescape_for_display(controldir.user_url, 'ascii')
        errors.BzrError.__init__(self, controldir=controldir, display_url=display_url)