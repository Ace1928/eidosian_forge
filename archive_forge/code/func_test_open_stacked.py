from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_open_stacked(self):
    b = _mod_branch.Branch.open(self.get_url('stacked'))
    rev = b.repository.get_revision(self.rev_base)
    self.assertEqual(1, len(self.connections))