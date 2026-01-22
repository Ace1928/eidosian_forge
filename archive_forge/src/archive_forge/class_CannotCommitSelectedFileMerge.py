from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
class CannotCommitSelectedFileMerge(BzrError):
    _fmt = 'Selected-file commit of merges is not supported yet: files %(files_str)s'

    def __init__(self, files):
        files_str = ', '.join(files)
        BzrError.__init__(self, files=files, files_str=files_str)