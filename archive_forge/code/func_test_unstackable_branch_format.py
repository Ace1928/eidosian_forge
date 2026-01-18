from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_unstackable_branch_format(self):
    format = 'foo'
    url = '/foo'
    error = _mod_branch.UnstackableBranchFormat(format, url)
    self.assertEqualDiff("The branch '/foo'(foo) is not a stackable format. You will need to upgrade the branch to permit branch stacking.", str(error))