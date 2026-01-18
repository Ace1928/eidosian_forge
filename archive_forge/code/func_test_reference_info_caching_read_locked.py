from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_reference_info_caching_read_locked(self):
    gets = []
    branch = self.create_branch_with_reference()
    branch.lock_read()
    self.addCleanup(branch.unlock)
    self.instrument_branch(branch, gets)
    branch.get_reference_info('path')
    branch.get_reference_info('path')
    self.assertEqual(1, len(gets))