from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
class InventoryDirectory(InventoryEntry):
    """A directory in an inventory."""
    __slots__ = ['children']
    kind = 'directory'

    def _check(self, checker, rev_id):
        """See InventoryEntry._check"""
        if self.name == '' and (not checker.rich_roots):
            return
        checker.add_pending_item(rev_id, ('texts', self.file_id, self.revision), b'text', b'da39a3ee5e6b4b0d3255bfef95601890afd80709')

    def copy(self):
        other = InventoryDirectory(self.file_id, self.name, self.parent_id)
        other.revision = self.revision
        return other

    def __init__(self, file_id, name, parent_id):
        super().__init__(file_id, name, parent_id)
        self.children = {}

    def sorted_children(self):
        return sorted(self.children.items())

    def kind_character(self):
        """See InventoryEntry.kind_character."""
        return '/'