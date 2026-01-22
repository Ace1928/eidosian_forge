from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
class InventoryLink(InventoryEntry):
    """A file in an inventory."""
    __slots__ = ['symlink_target']
    kind = 'symlink'

    def __init__(self, file_id, name, parent_id):
        super().__init__(file_id, name, parent_id)
        self.symlink_target = None

    def _check(self, checker, tree_revision_id):
        """See InventoryEntry._check"""
        if self.symlink_target is None:
            checker._report_items.append('symlink {%s} has no target in revision {%s}' % (self.file_id, tree_revision_id))
        checker.add_pending_item(tree_revision_id, ('texts', self.file_id, self.revision), b'text', b'da39a3ee5e6b4b0d3255bfef95601890afd80709')

    def copy(self):
        other = InventoryLink(self.file_id, self.name, self.parent_id)
        other.symlink_target = self.symlink_target
        other.revision = self.revision
        return other

    def detect_changes(self, old_entry):
        """See InventoryEntry.detect_changes."""
        text_modified = self.symlink_target != old_entry.symlink_target
        if text_modified:
            trace.mutter('    symlink target changed')
        meta_modified = False
        return (text_modified, meta_modified)

    def _diff(self, text_diff, from_label, tree, to_label, to_entry, to_tree, output_to, reverse=False):
        """See InventoryEntry._diff."""
        from breezy.diff import DiffSymlink
        old_target = self.symlink_target
        if to_entry is not None:
            new_target = to_entry.symlink_target
        else:
            new_target = None
        if not reverse:
            old_tree = tree
            new_tree = to_tree
        else:
            old_tree = to_tree
            new_tree = tree
            new_target, old_target = (old_target, new_target)
        differ = DiffSymlink(old_tree, new_tree, output_to)
        return differ.diff_symlink(old_target, new_target)

    def kind_character(self):
        """See InventoryEntry.kind_character."""
        return ''

    def _read_tree_state(self, path, work_tree):
        """See InventoryEntry._read_tree_state."""
        self.symlink_target = work_tree.get_symlink_target(work_tree.id2path(self.file_id), self.file_id)

    def _forget_tree_state(self):
        self.symlink_target = None

    def _unchanged(self, previous_ie):
        """See InventoryEntry._unchanged."""
        compatible = super()._unchanged(previous_ie)
        if self.symlink_target != previous_ie.symlink_target:
            compatible = False
        return compatible