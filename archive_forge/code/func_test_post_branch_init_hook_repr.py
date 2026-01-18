from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_post_branch_init_hook_repr(self):
    param_reprs = []
    _mod_branch.Branch.hooks.install_named_hook('post_branch_init', lambda params: param_reprs.append(repr(params)), None)
    branch = self.make_branch('a')
    self.assertLength(1, param_reprs)
    param_repr = param_reprs[0]
    self.assertStartsWith(param_repr, '<BranchInitHookParams of ')