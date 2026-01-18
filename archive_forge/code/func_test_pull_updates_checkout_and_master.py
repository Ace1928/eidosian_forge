from breezy import errors
from breezy.branch import BindingUnsupported, Branch
from breezy.controldir import ControlDir
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable
from breezy.tests.per_interbranch import TestCaseWithInterBranch
def test_pull_updates_checkout_and_master(self):
    """Pulling into a checkout updates the checkout and the master branch"""
    master_tree = self.make_from_branch_and_tree('master')
    master_tree.commit('master')
    checkout = master_tree.branch.create_checkout('checkout')
    try:
        other = self.sprout_to(master_tree.branch.controldir, 'other').open_workingtree()
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('lossless push between %r and %r not supported' % (self.branch_format_from, self.branch_format_to))
    rev2 = other.commit('other commit')
    try:
        checkout.branch.pull(other.branch)
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('lossless push between %r and %r not supported' % (self.branch_format_from, self.branch_format_to))
    self.assertEqual(rev2, checkout.branch.last_revision())
    self.assertEqual(rev2, master_tree.branch.last_revision())