import threading
from . import errors, trace, urlutils
from .branch import Branch
from .controldir import ControlDir, ControlDirFormat
from .transport import do_catching_redirections, get_transport
class AcceptAnythingPolicy(_BlacklistPolicy):
    """Accept anything, to make testing easier."""

    def __init__(self):
        super().__init__(True, set())