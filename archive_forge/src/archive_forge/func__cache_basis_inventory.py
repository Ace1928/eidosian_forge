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
def _cache_basis_inventory(self, new_revision):
    """Cache new_revision as the basis inventory."""
    try:
        lines = self.branch.repository._get_inventory_xml(new_revision)
        firstline = lines[0]
        if b'revision_id="' not in firstline or b'format="7"' not in firstline:
            inv = self.branch.repository._serializer.read_inventory_from_lines(lines, new_revision)
            lines = self._create_basis_xml_from_inventory(new_revision, inv)
        self._write_basis_inventory(lines)
    except (errors.NoSuchRevision, errors.RevisionNotPresent):
        pass