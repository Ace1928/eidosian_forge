from ... import controldir as _mod_controldir
from ... import errors, lockable_files
from ...branch import BindingUnsupported, BranchFormat, BranchWriteLockResult
from ...bzr.fullhistory import FullHistoryBzrBranch
from ...decorators import only_raises
from ...lock import LogicalLockResult
from ...trace import mutter
class BzrBranch4(FullHistoryBzrBranch):
    """Branch format 4."""

    def lock_write(self, token=None):
        """Lock the branch for write operations.

        :param token: A token to permit reacquiring a previously held and
            preserved lock.
        :return: A BranchWriteLockResult.
        """
        if not self.is_locked():
            self._note_lock('w')
        self.repository._warn_if_deprecated(self)
        self.repository.lock_write()
        try:
            return BranchWriteLockResult(self.unlock, self.control_files.lock_write(token=token))
        except:
            self.repository.unlock()
            raise

    def lock_read(self):
        """Lock the branch for read operations.

        :return: A breezy.lock.LogicalLockResult.
        """
        if not self.is_locked():
            self._note_lock('r')
        self.repository._warn_if_deprecated(self)
        self.repository.lock_read()
        try:
            self.control_files.lock_read()
            return LogicalLockResult(self.unlock)
        except:
            self.repository.unlock()
            raise

    @only_raises(errors.LockNotHeld, errors.LockBroken)
    def unlock(self):
        if self.control_files._lock_count == 2 and self.conf_store is not None:
            self.conf_store.save_changes()
        try:
            self.control_files.unlock()
        finally:
            self.repository.unlock()
            if not self.control_files.is_locked():
                self._clear_cached_state()

    def _get_checkout_format(self, lightweight=False):
        """Return the most suitable metadir for a checkout of this branch.
        """
        from ...bzr.bzrdir import BzrDirMetaFormat1
        from .repository import RepositoryFormat7
        format = BzrDirMetaFormat1()
        if lightweight:
            format.set_branch_format(self._format)
            format.repository_format = self.controldir._format.repository_format
        else:
            format.repository_format = RepositoryFormat7()
        return format

    def unbind(self):
        raise errors.UpgradeRequired(self.user_url)

    def bind(self, other):
        raise BindingUnsupported(self)

    def set_bound_location(self, location):
        raise NotImplementedError(self.set_bound_location)

    def get_bound_location(self):
        return None

    def update(self):
        return None

    def get_master_branch(self, possible_transports=None):
        return None