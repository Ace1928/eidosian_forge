from ..push import PushResult
from .errors import GitSmartRemoteNotSupported
def _lookup_revno(self, revid):
    from .branch import _quick_lookup_revno
    try:
        return _quick_lookup_revno(self.source_branch, self.target_branch, revid)
    except GitSmartRemoteNotSupported:
        return None