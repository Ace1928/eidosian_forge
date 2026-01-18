from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_use_fresh_values(self):
    copy = _mod_branch.Branch.open(self.branch.base)
    copy.lock_write()
    try:
        copy.get_config_stack().set('foo', 'bar')
    finally:
        copy.unlock()
    self.assertFalse(self.branch.is_locked())
    self.assertEqual(None, self.branch.get_config_stack().get('foo'))
    fresh = _mod_branch.Branch.open(self.branch.base)
    self.assertEqual('bar', fresh.get_config_stack().get('foo'))