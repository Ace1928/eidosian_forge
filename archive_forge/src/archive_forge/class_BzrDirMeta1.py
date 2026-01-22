import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
class BzrDirMeta1(BzrDir):
    """A .bzr meta version 1 control object.

    This is the first control object where the
    individual aspects are really split out: there are separate repository,
    workingtree and branch subdirectories and any subset of the three can be
    present within a BzrDir.
    """

    def _get_branch_path(self, name):
        """Obtain the branch path to use.

        This uses the API specified branch name first, and then falls back to
        the branch name specified in the URL. If neither of those is specified,
        it uses the default branch.

        :param name: Optional branch name to use
        :return: Relative path to branch
        """
        if name == '':
            return 'branch'
        return urlutils.join('branches', urlutils.escape(name))

    def _read_branch_list(self):
        """Read the branch list.

        :return: List of branch names.
        """
        try:
            f = self.control_transport.get('branch-list')
        except _mod_transport.NoSuchFile:
            return []
        ret = []
        try:
            for name in f:
                ret.append(name.rstrip(b'\n').decode('utf-8'))
        finally:
            f.close()
        return ret

    def _write_branch_list(self, branches):
        """Write out the branch list.

        :param branches: List of utf-8 branch names to write
        """
        self.transport.put_bytes('branch-list', b''.join([name.encode('utf-8') + b'\n' for name in branches]))

    def __init__(self, _transport, _format):
        super().__init__(_transport, _format)
        self.control_files = lockable_files.LockableFiles(self.control_transport, self._format._lock_file_name, self._format._lock_class)

    def can_convert_format(self):
        """See BzrDir.can_convert_format()."""
        return True

    def create_branch(self, name=None, repository=None, append_revisions_only=None):
        """See ControlDir.create_branch."""
        if name is None:
            name = self._get_selected_branch()
        return self._format.get_branch_format().initialize(self, name=name, repository=repository, append_revisions_only=append_revisions_only)

    def destroy_branch(self, name=None):
        """See ControlDir.destroy_branch."""
        if name is None:
            name = self._get_selected_branch()
        path = self._get_branch_path(name)
        if name != '':
            self.control_files.lock_write()
            try:
                branches = self._read_branch_list()
                try:
                    branches.remove(name)
                except ValueError:
                    raise errors.NotBranchError(name)
                self._write_branch_list(branches)
            finally:
                self.control_files.unlock()
        try:
            self.transport.delete_tree(path)
        except _mod_transport.NoSuchFile:
            raise errors.NotBranchError(path=urlutils.join(self.transport.base, path), controldir=self)

    def create_repository(self, shared=False):
        """See BzrDir.create_repository."""
        return self._format.repository_format.initialize(self, shared)

    def destroy_repository(self):
        """See BzrDir.destroy_repository."""
        try:
            self.transport.delete_tree('repository')
        except _mod_transport.NoSuchFile:
            raise errors.NoRepositoryPresent(self)

    def create_workingtree(self, revision_id=None, from_branch=None, accelerator_tree=None, hardlink=False):
        """See BzrDir.create_workingtree."""
        return self._format.workingtree_format.initialize(self, revision_id, from_branch=from_branch, accelerator_tree=accelerator_tree, hardlink=hardlink)

    def destroy_workingtree(self):
        """See BzrDir.destroy_workingtree."""
        wt = self.open_workingtree(recommend_upgrade=False)
        repository = wt.branch.repository
        empty = repository.revision_tree(_mod_revision.NULL_REVISION)
        wt.revert(old_tree=empty)
        self.destroy_workingtree_metadata()

    def destroy_workingtree_metadata(self):
        self.transport.delete_tree('checkout')

    def find_branch_format(self, name=None):
        """Find the branch 'format' for this bzrdir.

        This might be a synthetic object for e.g. RemoteBranch and SVN.
        """
        from .branch import BranchFormatMetadir
        return BranchFormatMetadir.find_format(self, name=name)

    def _get_mkdir_mode(self):
        """Figure out the mode to use when creating a bzrdir subdir."""
        temp_control = lockable_files.LockableFiles(self.transport, '', lockable_files.TransportLock)
        return temp_control._dir_mode

    def get_branch_reference(self, name=None):
        """See BzrDir.get_branch_reference()."""
        from .branch import BranchFormatMetadir
        format = BranchFormatMetadir.find_format(self, name=name)
        return format.get_reference(self, name=name)

    def set_branch_reference(self, target_branch, name=None):
        format = _mod_bzrbranch.BranchReferenceFormat()
        if self.control_url == target_branch.controldir.control_url and name == target_branch.name:
            raise controldir.BranchReferenceLoop(target_branch)
        return format.initialize(self, target_branch=target_branch, name=name)

    def get_branch_transport(self, branch_format, name=None):
        """See BzrDir.get_branch_transport()."""
        if name is None:
            name = self._get_selected_branch()
        path = self._get_branch_path(name)
        if branch_format is None:
            return self.transport.clone(path)
        try:
            branch_format.get_format_string()
        except NotImplementedError:
            raise errors.IncompatibleFormat(branch_format, self._format)
        if name != '':
            branches = self._read_branch_list()
            if name not in branches:
                self.control_files.lock_write()
                try:
                    branches = self._read_branch_list()
                    dirname = urlutils.dirname(name)
                    if dirname != '' and dirname in branches:
                        raise errors.ParentBranchExists(name)
                    child_branches = [b.startswith(name + '/') for b in branches]
                    if any(child_branches):
                        raise errors.AlreadyBranchError(name)
                    branches.append(name)
                    self._write_branch_list(branches)
                finally:
                    self.control_files.unlock()
        branch_transport = self.transport.clone(path)
        mode = self._get_mkdir_mode()
        branch_transport.create_prefix(mode=mode)
        try:
            self.transport.mkdir(path, mode=mode)
        except _mod_transport.FileExists:
            pass
        return self.transport.clone(path)

    def get_repository_transport(self, repository_format):
        """See BzrDir.get_repository_transport()."""
        if repository_format is None:
            return self.transport.clone('repository')
        try:
            repository_format.get_format_string()
        except NotImplementedError:
            raise errors.IncompatibleFormat(repository_format, self._format)
        try:
            self.transport.mkdir('repository', mode=self._get_mkdir_mode())
        except _mod_transport.FileExists:
            pass
        return self.transport.clone('repository')

    def get_workingtree_transport(self, workingtree_format):
        """See BzrDir.get_workingtree_transport()."""
        if workingtree_format is None:
            return self.transport.clone('checkout')
        try:
            workingtree_format.get_format_string()
        except NotImplementedError:
            raise errors.IncompatibleFormat(workingtree_format, self._format)
        try:
            self.transport.mkdir('checkout', mode=self._get_mkdir_mode())
        except _mod_transport.FileExists:
            pass
        return self.transport.clone('checkout')

    def branch_names(self):
        """See ControlDir.branch_names."""
        ret = []
        try:
            self.get_branch_reference()
        except errors.NotBranchError:
            pass
        else:
            ret.append('')
        ret.extend(self._read_branch_list())
        return ret

    def get_branches(self):
        """See ControlDir.get_branches."""
        ret = {}
        try:
            ret[''] = self.open_branch(name='')
        except (errors.NotBranchError, errors.NoRepositoryPresent):
            pass
        for name in self._read_branch_list():
            ret[name] = self.open_branch(name=name)
        return ret

    def has_workingtree(self):
        """Tell if this bzrdir contains a working tree.

        Note: if you're going to open the working tree, you should just go
        ahead and try, and not ask permission first.
        """
        from .workingtree import WorkingTreeFormatMetaDir
        try:
            WorkingTreeFormatMetaDir.find_format_string(self)
        except errors.NoWorkingTree:
            return False
        return True

    def needs_format_conversion(self, format):
        """See BzrDir.needs_format_conversion()."""
        if not isinstance(self._format, format.__class__) or self._format.get_format_string() != format.get_format_string():
            return True
        try:
            if not isinstance(self.open_repository()._format, format.repository_format.__class__):
                return True
        except errors.NoRepositoryPresent:
            pass
        for branch in self.list_branches():
            if not isinstance(branch._format, format.get_branch_format().__class__):
                return True
        try:
            my_wt = self.open_workingtree(recommend_upgrade=False)
            if not isinstance(my_wt._format, format.workingtree_format.__class__):
                return True
        except (errors.NoWorkingTree, errors.NotLocalUrl):
            pass
        return False

    def open_branch(self, name=None, unsupported=False, ignore_fallbacks=False, possible_transports=None):
        """See ControlDir.open_branch."""
        if name is None:
            name = self._get_selected_branch()
        format = self.find_branch_format(name=name)
        format.check_support_status(unsupported)
        if possible_transports is None:
            possible_transports = []
        else:
            possible_transports = list(possible_transports)
        possible_transports.append(self.root_transport)
        return format.open(self, name=name, _found=True, ignore_fallbacks=ignore_fallbacks, possible_transports=possible_transports)

    def open_repository(self, unsupported=False):
        """See BzrDir.open_repository."""
        from .repository import RepositoryFormatMetaDir
        format = RepositoryFormatMetaDir.find_format(self)
        format.check_support_status(unsupported)
        return format.open(self, _found=True)

    def open_workingtree(self, unsupported=False, recommend_upgrade=True):
        """See BzrDir.open_workingtree."""
        from .workingtree import WorkingTreeFormatMetaDir
        format = WorkingTreeFormatMetaDir.find_format(self)
        format.check_support_status(unsupported, recommend_upgrade, basedir=self.root_transport.base)
        return format.open(self, _found=True)

    def _get_config(self):
        return config.TransportConfig(self.transport, 'control.conf')