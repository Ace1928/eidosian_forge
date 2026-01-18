from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_set_delays_write_when_branch_is_locked(self):
    self.branch.lock_write()
    self.addCleanup(self.branch.unlock)
    self.branch.get_config_stack().set('foo', 'bar')
    copy = _mod_branch.Branch.open(self.branch.base)
    result = copy.get_config_stack().get('foo')
    self.assertIs(None, result)