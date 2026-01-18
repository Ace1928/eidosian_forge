from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_open_opens_stacked_reference(self):
    branch = self.make_branch('a', format=self.get_format_name())
    target = self.make_branch_and_tree('b', format=self.get_format_name())
    branch.set_stacked_on_url(target.branch.base)
    branch = branch.controldir.open_branch()
    revid = target.commit('foo')
    self.assertTrue(branch.repository.has_revision(revid))