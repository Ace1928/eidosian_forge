from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_set_from_config_stack_get_from_config(self):
    self.branch.lock_write()
    self.addCleanup(self.branch.unlock)
    self.branch.get_config_stack().set('foo', 'bar')
    self.assertEqual(None, self.branch.get_config().get_user_option('foo'))