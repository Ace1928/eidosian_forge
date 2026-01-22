from ... import controldir as _mod_controldir
from ... import errors, lockable_files
from ...branch import BindingUnsupported, BranchFormat, BranchWriteLockResult
from ...bzr.fullhistory import FullHistoryBzrBranch
from ...decorators import only_raises
from ...lock import LogicalLockResult
from ...trace import mutter
class BzrBranchFormat4(BranchFormat):
    """Bzr branch format 4.

    This format has:
     - a revision-history file.
     - a branch-lock lock file [ to be shared with the bzrdir ]

    It does not support binding.
    """

    def initialize(self, a_controldir, name=None, repository=None, append_revisions_only=None):
        """Create a branch of this format in a_controldir.

        :param a_controldir: The bzrdir to initialize the branch in
        :param name: Name of colocated branch to create, if any
        :param repository: Repository for this branch (unused)
        """
        if append_revisions_only:
            raise errors.UpgradeRequired(a_controldir.user_url)
        if repository is not None:
            raise NotImplementedError('initialize(repository=<not None>) on {!r}'.format(self))
        if not [isinstance(a_controldir._format, format) for format in self._compatible_bzrdirs]:
            raise errors.IncompatibleFormat(self, a_controldir._format)
        utf8_files = [('revision-history', b''), ('branch-name', b'')]
        mutter('creating branch %r in %s', self, a_controldir.user_url)
        branch_transport = a_controldir.get_branch_transport(self, name=name)
        control_files = lockable_files.LockableFiles(branch_transport, 'branch-lock', lockable_files.TransportLock)
        control_files.create_lock()
        try:
            control_files.lock_write()
        except errors.LockContention:
            lock_taken = False
        else:
            lock_taken = True
        try:
            for filename, content in utf8_files:
                branch_transport.put_bytes(filename, content, mode=a_controldir._get_file_mode())
        finally:
            if lock_taken:
                control_files.unlock()
        branch = self.open(a_controldir, name, _found=True, found_repository=None)
        self._run_post_branch_init_hooks(a_controldir, name, branch)
        return branch

    def __init__(self):
        super().__init__()
        from .bzrdir import BzrDirFormat4, BzrDirFormat5, BzrDirFormat6
        self._matchingcontroldir = BzrDirFormat6()
        self._compatible_bzrdirs = [BzrDirFormat4, BzrDirFormat5, BzrDirFormat6]

    def network_name(self):
        """The network name for this format is the control dirs disk label."""
        return self._matchingcontroldir.get_format_string()

    def get_format_description(self):
        return 'Branch format 4'

    def open(self, a_controldir, name=None, _found=False, ignore_fallbacks=False, found_repository=None, possible_transports=None):
        """See BranchFormat.open()."""
        if name is None:
            name = a_controldir._get_selected_branch()
        if name != '':
            raise _mod_controldir.NoColocatedBranchSupport(self)
        if not _found:
            raise NotImplementedError
        if found_repository is None:
            found_repository = a_controldir.open_repository()
        return BzrBranch4(_format=self, _control_files=a_controldir._control_files, a_controldir=a_controldir, name=name, _repository=found_repository, possible_transports=possible_transports)

    def __str__(self):
        return 'Bazaar-NG branch format 4'

    def supports_leaving_lock(self):
        return False
    supports_reference_locations = False