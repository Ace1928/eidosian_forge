from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
def _set_progress_stage(self, name, counter=False):
    """Set the progress stage and emit an update to the progress bar."""
    self.pb_stage_name = name
    self.pb_stage_count += 1
    if counter:
        self.pb_entries_count = 0
    else:
        self.pb_entries_count = None
    self._emit_progress()