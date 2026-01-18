from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def test_report_unchanged(self):
    r = _mod_branch.PullResult()
    r.old_revid = b'same-revid'
    r.new_revid = b'same-revid'
    f = StringIO()
    r.report(f)
    self.assertEqual('No revisions or tags to pull.\n', f.getvalue())