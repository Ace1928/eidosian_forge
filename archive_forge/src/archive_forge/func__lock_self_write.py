import os
from io import BytesIO
from ..lazy_import import lazy_import
import contextlib
import errno
import stat
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import revision as _mod_revision
from ..lock import LogicalLockResult
from ..lockable_files import LockableFiles
from ..lockdir import LockDir
from ..mutabletree import BadReferenceTarget, MutableTree
from ..osutils import file_kind, isdir, pathjoin, realpath, safe_unicode
from ..transport import NoSuchFile, get_transport_from_path
from ..transport.local import LocalTransport
from ..tree import FileTimestampUnavailable, InterTree, MissingNestedTree
from ..workingtree import WorkingTree
from . import dirstate
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import (InterInventoryTree, InventoryRevisionTree,
from .workingtree import InventoryWorkingTree, WorkingTreeFormatMetaDir
def _lock_self_write(self):
    """This should be called after the branch is locked."""
    try:
        self._control_files.lock_write()
        try:
            state = self.current_dirstate()
            if not state._lock_token:
                state.lock_write()
            self._repo_supports_tree_reference = getattr(self.branch.repository._format, 'supports_tree_reference', False)
        except BaseException:
            self._control_files.unlock()
            raise
    except BaseException:
        self.branch.unlock()
        raise
    return LogicalLockResult(self.unlock)