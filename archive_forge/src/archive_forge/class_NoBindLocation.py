from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
class NoBindLocation(BzrDirError):
    _fmt = 'No location could be found to bind to at %(display_url)s.'