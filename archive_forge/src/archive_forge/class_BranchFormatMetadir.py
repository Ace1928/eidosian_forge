from io import BytesIO
from typing import TYPE_CHECKING, Optional, Union
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from .. import errors, lockable_files
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from .. import urlutils
from ..branch import (Branch, BranchFormat, BranchWriteLockResult,
from ..controldir import ControlDir
from ..decorators import only_raises
from ..lock import LogicalLockResult, _RelockDebugMixin
from ..trace import mutter
from . import bzrdir, rio
from .repository import MetaDirRepository
class BranchFormatMetadir(bzrdir.BzrFormat, BranchFormat):
    """Base class for branch formats that live in meta directories.
    """

    def __init__(self):
        BranchFormat.__init__(self)
        bzrdir.BzrFormat.__init__(self)

    @classmethod
    def find_format(klass, controldir, name=None):
        """Return the format for the branch object in controldir."""
        try:
            transport = controldir.get_branch_transport(None, name=name)
        except _mod_transport.NoSuchFile as exc:
            raise errors.NotBranchError(path=name, controldir=controldir) from exc
        try:
            format_string = transport.get_bytes('format')
        except _mod_transport.NoSuchFile as exc:
            raise errors.NotBranchError(path=transport.base, controldir=controldir) from exc
        return klass._find_format(format_registry, 'branch', format_string)

    def _branch_class(self):
        """What class to instantiate on open calls."""
        raise NotImplementedError(self._branch_class)

    def _get_initial_config(self, append_revisions_only=None):
        if append_revisions_only:
            return b'append_revisions_only = True\n'
        else:
            return b''

    def _initialize_helper(self, a_controldir, utf8_files, name=None, repository=None):
        """Initialize a branch in a control dir, with specified files

        :param a_controldir: The bzrdir to initialize the branch in
        :param utf8_files: The files to create as a list of
            (filename, content) tuples
        :param name: Name of colocated branch to create, if any
        :return: a branch in this format
        """
        if name is None:
            name = a_controldir._get_selected_branch()
        mutter('creating branch %r in %s', self, a_controldir.user_url)
        branch_transport = a_controldir.get_branch_transport(self, name=name)
        control_files = lockable_files.LockableFiles(branch_transport, 'lock', lockdir.LockDir)
        control_files.create_lock()
        control_files.lock_write()
        try:
            utf8_files += [('format', self.as_string())]
            for filename, content in utf8_files:
                branch_transport.put_bytes(filename, content, mode=a_controldir._get_file_mode())
        finally:
            control_files.unlock()
        branch = self.open(a_controldir, name, _found=True, found_repository=repository)
        self._run_post_branch_init_hooks(a_controldir, name, branch)
        return branch

    def open(self, a_controldir, name=None, _found=False, ignore_fallbacks=False, found_repository=None, possible_transports=None):
        """See BranchFormat.open()."""
        if name is None:
            name = a_controldir._get_selected_branch()
        if not _found:
            format = BranchFormatMetadir.find_format(a_controldir, name=name)
            if format.__class__ != self.__class__:
                raise AssertionError('wrong format %r found for %r' % (format, self))
        transport = a_controldir.get_branch_transport(None, name=name)
        try:
            control_files = lockable_files.LockableFiles(transport, 'lock', lockdir.LockDir)
            if found_repository is None:
                found_repository = a_controldir.find_repository()
            return self._branch_class()(_format=self, _control_files=control_files, name=name, a_controldir=a_controldir, _repository=found_repository, ignore_fallbacks=ignore_fallbacks, possible_transports=possible_transports)
        except _mod_transport.NoSuchFile as exc:
            raise errors.NotBranchError(path=transport.base, controldir=a_controldir) from exc

    @property
    def _matchingcontroldir(self):
        ret = bzrdir.BzrDirMetaFormat1()
        ret.set_branch_format(self)
        return ret

    def supports_tags(self):
        return True

    def supports_leaving_lock(self):
        return True

    def check_support_status(self, allow_unsupported, recommend_upgrade=True, basedir=None):
        BranchFormat.check_support_status(self, allow_unsupported=allow_unsupported, recommend_upgrade=recommend_upgrade, basedir=basedir)
        bzrdir.BzrFormat.check_support_status(self, allow_unsupported=allow_unsupported, recommend_upgrade=recommend_upgrade, basedir=basedir)