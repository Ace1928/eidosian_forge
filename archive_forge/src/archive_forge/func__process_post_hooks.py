from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
def _process_post_hooks(self, old_revno, new_revno):
    """Process any registered post commit hooks."""
    self._set_progress_stage('Running post_commit hooks')
    post_commit = self.config_stack.get('post_commit')
    if post_commit is not None:
        hooks = post_commit.split(' ')
        for hook in hooks:
            result = eval(hook + '(branch, rev_id)', {'branch': self.branch, 'breezy': breezy, 'rev_id': self.rev_id})
    self._process_hooks('post_commit', old_revno, new_revno)