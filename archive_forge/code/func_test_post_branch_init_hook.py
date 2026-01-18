from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_post_branch_init_hook(self):
    calls = []
    _mod_branch.Branch.hooks.install_named_hook('post_branch_init', calls.append, None)
    self.assertLength(0, calls)
    branch = self.make_branch('a')
    self.assertLength(1, calls)
    params = calls[0]
    self.assertIsInstance(params, _mod_branch.BranchInitHookParams)
    self.assertTrue(hasattr(params, 'controldir'))
    self.assertTrue(hasattr(params, 'branch'))