from ..push import PushResult
from .errors import GitSmartRemoteNotSupported
class GitPushResult(PushResult):

    def _lookup_revno(self, revid):
        from .branch import _quick_lookup_revno
        try:
            return _quick_lookup_revno(self.source_branch, self.target_branch, revid)
        except GitSmartRemoteNotSupported:
            return None

    @property
    def old_revno(self):
        return self._lookup_revno(self.old_revid)

    @property
    def new_revno(self):
        return self._lookup_revno(self.new_revid)