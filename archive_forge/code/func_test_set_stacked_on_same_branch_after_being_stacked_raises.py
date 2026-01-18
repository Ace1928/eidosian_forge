from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_set_stacked_on_same_branch_after_being_stacked_raises(self):
    branch = self.make_branch('branch')
    target = self.make_branch('target')
    try:
        branch.set_stacked_on_url('../target')
    except unstackable_format_errors:
        self.assertRaises(unstackable_format_errors, branch.get_stacked_on_url)
        return
    self.assertRaises(errors.UnstackableLocationError, branch.set_stacked_on_url, '../branch')
    self.assertEqual('../target', branch.get_stacked_on_url())