import errno
import itertools
import operator
import os
import stat
import sys
from bisect import bisect_left
from collections import deque
from io import BytesIO
import breezy
from .. import lazy_import
from . import bzrdir
import contextlib
from breezy import (
from breezy.bzr import (
from .. import errors, osutils
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from ..controldir import ControlDir
from ..lock import LogicalLockResult
from ..trace import mutter, note
from ..tree import (MissingNestedTree, TreeDirectory, TreeEntry, TreeFile,
from ..workingtree import WorkingTree, WorkingTreeFormat, format_registry
from .inventorytree import InventoryRevisionTree, MutableInventoryTree
def _rollback_move(self, moved):
    """Try to rollback a previous move in case of an filesystem error."""
    for entry in moved:
        try:
            self._move_entry(WorkingTree._RenameEntry(entry.to_rel, entry.from_id, entry.to_tail, entry.to_parent_id, entry.from_rel, entry.from_tail, entry.from_parent_id, entry.only_change_inv))
        except errors.BzrMoveFailedError as e:
            raise errors.BzrMoveFailedError('', '', "Rollback failed. The working tree is in an inconsistent state. Please consider doing a 'bzr revert'. Error message is: %s" % e)