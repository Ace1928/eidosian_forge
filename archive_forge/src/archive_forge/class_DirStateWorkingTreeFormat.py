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
class DirStateWorkingTreeFormat(WorkingTreeFormatMetaDir):
    missing_parent_conflicts = True
    supports_versioned_directories = True
    _lock_class = LockDir
    _lock_file_name = 'lock'

    def _open_control_files(self, a_controldir):
        transport = a_controldir.get_workingtree_transport(None)
        return LockableFiles(transport, self._lock_file_name, self._lock_class)

    def initialize(self, a_controldir, revision_id=None, from_branch=None, accelerator_tree=None, hardlink=False):
        """See WorkingTreeFormat.initialize().

        :param revision_id: allows creating a working tree at a different
            revision than the branch is at.
        :param accelerator_tree: A tree which can be used for retrieving file
            contents more quickly than the revision tree, i.e. a workingtree.
            The revision tree will be used for cases where accelerator_tree's
            content is different.
        :param hardlink: If true, hard-link files from accelerator_tree,
            where possible.

        These trees get an initial random root id, if their repository supports
        rich root data, TREE_ROOT otherwise.
        """
        if not isinstance(a_controldir.transport, LocalTransport):
            raise errors.NotLocalUrl(a_controldir.transport.base)
        transport = a_controldir.get_workingtree_transport(self)
        control_files = self._open_control_files(a_controldir)
        control_files.create_lock()
        control_files.lock_write()
        transport.put_bytes('format', self.as_string(), mode=a_controldir._get_file_mode())
        if from_branch is not None:
            branch = from_branch
        else:
            branch = a_controldir.open_branch()
        if revision_id is None:
            revision_id = branch.last_revision()
        local_path = transport.local_abspath('dirstate')
        state = dirstate.DirState.initialize(local_path)
        state.unlock()
        del state
        wt = self._tree_class(a_controldir.root_transport.local_abspath('.'), branch, _format=self, _controldir=a_controldir, _control_files=control_files)
        wt._new_tree()
        wt.lock_tree_write()
        try:
            self._init_custom_control_files(wt)
            if revision_id in (None, _mod_revision.NULL_REVISION):
                if branch.repository.supports_rich_root():
                    wt._set_root_id(generate_ids.gen_root_id())
                else:
                    wt._set_root_id(ROOT_ID)
                wt.flush()
            basis = None
            if accelerator_tree is not None:
                try:
                    basis = accelerator_tree.revision_tree(revision_id)
                except errors.NoSuchRevision:
                    pass
            if basis is None:
                basis = branch.repository.revision_tree(revision_id)
            if revision_id == _mod_revision.NULL_REVISION:
                parents_list = []
            else:
                parents_list = [(revision_id, basis)]
            with basis.lock_read():
                wt.set_parent_trees(parents_list, allow_leftmost_as_ghost=True)
                wt.flush()
                basis_root_id = basis.path2id('')
                if basis_root_id is not None:
                    wt._set_root_id(basis_root_id)
                    wt.flush()
                if wt.supports_content_filtering():
                    delta_from_tree = False
                else:
                    delta_from_tree = True
                bzr_transform.build_tree(basis, wt, accelerator_tree, hardlink=hardlink, delta_from_tree=delta_from_tree)
                for hook in MutableTree.hooks['post_build_tree']:
                    hook(wt)
        finally:
            control_files.unlock()
            wt.unlock()
        return wt

    def _init_custom_control_files(self, wt):
        """Subclasses with custom control files should override this method.

        The working tree and control files are locked for writing when this
        method is called.

        :param wt: the WorkingTree object
        """

    def open(self, a_controldir, _found=False):
        """Return the WorkingTree object for a_controldir

        _found is a private parameter, do not use it. It is used to indicate
               if format probing has already been done.
        """
        if not _found:
            raise NotImplementedError
        if not isinstance(a_controldir.transport, LocalTransport):
            raise errors.NotLocalUrl(a_controldir.transport.base)
        wt = self._open(a_controldir, self._open_control_files(a_controldir))
        return wt

    def _open(self, a_controldir, control_files):
        """Open the tree itself.

        :param a_controldir: the dir for the tree.
        :param control_files: the control files for the tree.
        """
        return self._tree_class(a_controldir.root_transport.local_abspath('.'), branch=a_controldir.open_branch(), _format=self, _controldir=a_controldir, _control_files=control_files)

    def __get_matchingcontroldir(self):
        return self._get_matchingcontroldir()

    def _get_matchingcontroldir(self):
        """Overrideable method to get a bzrdir for testing."""
        return controldir.format_registry.make_controldir('development-subtree')
    _matchingcontroldir = property(__get_matchingcontroldir)