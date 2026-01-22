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
def _set_inventory(self, inv, dirty):
    """Set the internal cached inventory.

        :param inv: The inventory to set.
        :param dirty: A boolean indicating whether the inventory is the same
            logical inventory as whats on disk. If True the inventory is not
            the same and should be written to disk or data will be lost, if
            False then the inventory is the same as that on disk and any
            serialisation would be unneeded overhead.
        """
    self._inventory = inv
    self._inventory_is_modified = dirty