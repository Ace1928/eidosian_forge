from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
class TreeReference(InventoryEntry):
    __slots__ = ['reference_revision']
    kind = 'tree-reference'

    def __init__(self, file_id, name, parent_id, revision=None, reference_revision=None):
        InventoryEntry.__init__(self, file_id, name, parent_id)
        self.revision = revision
        self.reference_revision = reference_revision

    def copy(self):
        return TreeReference(self.file_id, self.name, self.parent_id, self.revision, self.reference_revision)

    def _read_tree_state(self, path, work_tree):
        """Populate fields in the inventory entry from the given tree.
        """
        self.reference_revision = work_tree.get_reference_revision(path, self.file_id)

    def _forget_tree_state(self):
        self.reference_revision = None

    def _unchanged(self, previous_ie):
        """See InventoryEntry._unchanged."""
        compatible = super()._unchanged(previous_ie)
        if self.reference_revision != previous_ie.reference_revision:
            compatible = False
        return compatible

    def kind_character(self):
        """See InventoryEntry.kind_character."""
        return '+'