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
class ContentFilteringDirStateWorkingTree(DirStateWorkingTree):
    """Dirstate working tree that supports content filtering.

    The dirstate holds the hash and size of the canonical form of the file,
    and most methods must return that.
    """

    def _file_content_summary(self, path, stat_result):
        dirstate_sha1 = self._dirstate.sha1_from_stat(path, stat_result)
        executable = self._is_executable_from_path_and_stat(path, stat_result)
        return ('file', None, executable, dirstate_sha1)