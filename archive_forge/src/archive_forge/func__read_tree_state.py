from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _read_tree_state(self, path, work_tree):
    """Populate fields in the inventory entry from the given tree.
        """
    self.reference_revision = work_tree.get_reference_revision(path, self.file_id)