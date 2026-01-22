from io import BytesIO
from ... import errors, lockable_files
from ...bzr.bzrdir import BzrDir, BzrDirFormat, BzrDirMetaFormat1
from ...controldir import (ControlDir, Converter, MustHaveWorkingTree,
from ...i18n import gettext
from ...lazy_import import lazy_import
from ...transport import NoSuchFile, get_transport, local
import os
from breezy import (
from breezy.bzr import (
from breezy.plugins.weave_fmt.store.versioned import VersionedFileStore
from breezy.transactions import WriteTransaction
from breezy.plugins.weave_fmt import xml4
class BzrDirPreSplitOut(BzrDir):
    """A common class for the all-in-one formats."""

    def __init__(self, _transport, _format):
        """See ControlDir.__init__."""
        super().__init__(_transport, _format)
        self._control_files = lockable_files.LockableFiles(self.get_branch_transport(None), self._format._lock_file_name, self._format._lock_class)

    def break_lock(self):
        """Pre-splitout bzrdirs do not suffer from stale locks."""
        raise NotImplementedError(self.break_lock)

    def cloning_metadir(self, require_stacking=False):
        """Produce a metadir suitable for cloning with."""
        if require_stacking:
            return format_registry.make_controldir('1.6')
        return self._format.__class__()

    def clone(self, url, revision_id=None, force_new_repo=False, preserve_stacking=False, tag_selector=None):
        """See ControlDir.clone().

        force_new_repo has no effect, since this family of formats always
        require a new repository.
        preserve_stacking has no effect, since no source branch using this
        family of formats can be stacked, so there is no stacking to preserve.
        """
        self._make_tail(url)
        result = self._format._initialize_for_clone(url)
        self.open_repository().clone(result, revision_id=revision_id)
        from_branch = self.open_branch()
        from_branch.clone(result, revision_id=revision_id, tag_selector=tag_selector)
        try:
            tree = self.open_workingtree()
        except errors.NotLocalUrl:
            result._init_workingtree()
        else:
            tree.clone(result)
        return result

    def create_branch(self, name=None, repository=None, append_revisions_only=None):
        """See ControlDir.create_branch."""
        if repository is not None:
            raise NotImplementedError('create_branch(repository=<not None>) on {!r}'.format(self))
        return self._format.get_branch_format().initialize(self, name=name, append_revisions_only=append_revisions_only)

    def destroy_branch(self, name=None):
        """See ControlDir.destroy_branch."""
        raise errors.UnsupportedOperation(self.destroy_branch, self)

    def create_repository(self, shared=False):
        """See ControlDir.create_repository."""
        if shared:
            raise errors.IncompatibleFormat('shared repository', self._format)
        return self.open_repository()

    def destroy_repository(self):
        """See ControlDir.destroy_repository."""
        raise errors.UnsupportedOperation(self.destroy_repository, self)

    def create_workingtree(self, revision_id=None, from_branch=None, accelerator_tree=None, hardlink=False):
        """See ControlDir.create_workingtree."""
        if hardlink:
            warning("can't support hardlinked working trees in %r" % (self,))
        try:
            result = self.open_workingtree(recommend_upgrade=False)
        except NoSuchFile:
            result = self._init_workingtree()
        if revision_id is not None:
            if revision_id == _mod_revision.NULL_REVISION:
                result.set_parent_ids([])
            else:
                result.set_parent_ids([revision_id])
        return result

    def _init_workingtree(self):
        from .workingtree import WorkingTreeFormat2
        try:
            return WorkingTreeFormat2().initialize(self)
        except errors.NotLocalUrl:
            return WorkingTreeFormat2()._stub_initialize_on_transport(self.transport, self._control_files._file_mode)

    def destroy_workingtree(self):
        """See ControlDir.destroy_workingtree."""
        raise errors.UnsupportedOperation(self.destroy_workingtree, self)

    def destroy_workingtree_metadata(self):
        """See ControlDir.destroy_workingtree_metadata."""
        raise errors.UnsupportedOperation(self.destroy_workingtree_metadata, self)

    def get_branch_transport(self, branch_format, name=None):
        """See BzrDir.get_branch_transport()."""
        if name:
            raise NoColocatedBranchSupport(self)
        if branch_format is None:
            return self.transport
        try:
            branch_format.get_format_string()
        except NotImplementedError:
            return self.transport
        raise errors.IncompatibleFormat(branch_format, self._format)

    def get_repository_transport(self, repository_format):
        """See BzrDir.get_repository_transport()."""
        if repository_format is None:
            return self.transport
        try:
            repository_format.get_format_string()
        except NotImplementedError:
            return self.transport
        raise errors.IncompatibleFormat(repository_format, self._format)

    def get_workingtree_transport(self, workingtree_format):
        """See BzrDir.get_workingtree_transport()."""
        if workingtree_format is None:
            return self.transport
        try:
            workingtree_format.get_format_string()
        except NotImplementedError:
            return self.transport
        raise errors.IncompatibleFormat(workingtree_format, self._format)

    def needs_format_conversion(self, format):
        """See ControlDir.needs_format_conversion()."""
        return not isinstance(self._format, format.__class__)

    def open_branch(self, name=None, unsupported=False, ignore_fallbacks=False, possible_transports=None):
        """See ControlDir.open_branch."""
        from .branch import BzrBranchFormat4
        format = BzrBranchFormat4()
        format.check_support_status(unsupported)
        return format.open(self, name, _found=True, possible_transports=possible_transports)

    def sprout(self, url, revision_id=None, force_new_repo=False, recurse=None, possible_transports=None, accelerator_tree=None, hardlink=False, stacked=False, create_tree_if_local=True, source_branch=None):
        """See ControlDir.sprout()."""
        if source_branch is not None:
            my_branch = self.open_branch()
            if source_branch.base != my_branch.base:
                raise AssertionError('source branch %r is not within %r with branch %r' % (source_branch, self, my_branch))
        if stacked:
            raise _mod_branch.UnstackableBranchFormat(self._format, self.root_transport.base)
        if not create_tree_if_local:
            raise MustHaveWorkingTree(self._format, self.root_transport.base)
        from .workingtree import WorkingTreeFormat2
        self._make_tail(url)
        result = self._format._initialize_for_clone(url)
        try:
            self.open_repository().clone(result, revision_id=revision_id)
        except errors.NoRepositoryPresent:
            pass
        try:
            self.open_branch().sprout(result, revision_id=revision_id)
        except errors.NotBranchError:
            pass
        WorkingTreeFormat2().initialize(result, accelerator_tree=accelerator_tree, hardlink=hardlink)
        return result

    def set_branch_reference(self, target_branch, name=None):
        from ...bzr.branch import BranchReferenceFormat
        if name is not None:
            raise NoColocatedBranchSupport(self)
        raise errors.IncompatibleFormat(BranchReferenceFormat, self._format)