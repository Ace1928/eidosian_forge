from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
class NullCommitReporter:
    """I report on progress of a commit."""

    def started(self, revno, revid, location):
        pass

    def snapshot_change(self, change, path):
        pass

    def completed(self, revno, rev_id):
        pass

    def deleted(self, path):
        pass

    def missing(self, path):
        pass

    def renamed(self, change, old_path, new_path):
        pass

    def is_verbose(self):
        return False