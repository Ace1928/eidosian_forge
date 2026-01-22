from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
def _filter_iter_changes(self, iter_changes):
    """Process iter_changes.

        This method reports on the changes in iter_changes to the user, and
        converts 'missing' entries in the iter_changes iterator to 'deleted'
        entries. 'missing' entries have their

        :param iter_changes: An iter_changes to process.
        :return: A generator of changes.
        """
    reporter = self.reporter
    report_changes = reporter.is_verbose()
    deleted_paths = []
    for change in iter_changes:
        if report_changes:
            old_path = change.path[0]
            new_path = change.path[1]
            versioned = change.versioned[1]
        kind = change.kind[1]
        versioned = change.versioned[1]
        if kind is None and versioned:
            if report_changes:
                reporter.missing(new_path)
            if change.kind[0] == 'symlink' and (not self.work_tree.supports_symlinks()):
                trace.warning('Ignoring "%s" as symlinks are not supported on this filesystem.' % (change.path[0],))
                continue
            deleted_paths.append(change.path[1])
            change = change.discard_new()
            new_path = change.path[1]
            versioned = False
        elif kind == 'tree-reference':
            if self.recursive == 'down':
                self._commit_nested_tree(change.path[1])
        if change.versioned[0] or change.versioned[1]:
            yield change
            if report_changes:
                if new_path is None:
                    reporter.deleted(old_path)
                elif old_path is None:
                    reporter.snapshot_change(gettext('added'), new_path)
                elif old_path != new_path:
                    reporter.renamed(gettext('renamed'), old_path, new_path)
                elif new_path or self.work_tree.branch.repository._format.rich_root_data:
                    reporter.snapshot_change(gettext('modified'), new_path)
        self._next_progress_entry()
    self.deleted_paths = deleted_paths