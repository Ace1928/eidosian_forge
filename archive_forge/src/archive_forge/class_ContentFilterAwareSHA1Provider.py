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
class ContentFilterAwareSHA1Provider(dirstate.SHA1Provider):

    def __init__(self, tree):
        self.tree = tree

    def sha1(self, abspath):
        """See dirstate.SHA1Provider.sha1()."""
        filters = self.tree._content_filter_stack(self.tree.relpath(osutils.safe_unicode(abspath)))
        return _mod_filters.internal_size_sha_file_byname(abspath, filters)[1]

    def stat_and_sha1(self, abspath):
        """See dirstate.SHA1Provider.stat_and_sha1()."""
        filters = self.tree._content_filter_stack(self.tree.relpath(osutils.safe_unicode(abspath)))
        with open(abspath, 'rb', 65000) as file_obj:
            statvalue = os.fstat(file_obj.fileno())
            if filters:
                file_obj, size = _mod_filters.filtered_input_file(file_obj, filters)
                statvalue = _mod_filters.FilteredStat(statvalue, size)
            sha1 = osutils.size_sha_file(file_obj)[1]
        return (statvalue, sha1)