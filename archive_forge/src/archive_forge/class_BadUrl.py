import threading
from . import errors, trace, urlutils
from .branch import Branch
from .controldir import ControlDir, ControlDirFormat
from .transport import do_catching_redirections, get_transport
class BadUrl(errors.BzrError):
    _fmt = 'Tried to access a branch from bad URL %(url)s.'