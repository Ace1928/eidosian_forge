from ... import controldir as _mod_controldir
from ... import errors, lockable_files
from ...branch import BindingUnsupported, BranchFormat, BranchWriteLockResult
from ...bzr.fullhistory import FullHistoryBzrBranch
from ...decorators import only_raises
from ...lock import LogicalLockResult
from ...trace import mutter
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