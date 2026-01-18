from breezy import errors
from breezy.branch import BindingUnsupported, Branch
from breezy.controldir import ControlDir
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable
from breezy.tests.per_interbranch import TestCaseWithInterBranch
def test_pull_returns_result(self):
    parent = self.make_from_branch_and_tree('parent')
    p1 = parent.commit('1st post')
    try:
        mine = self.sprout_to(parent.controldir, 'mine').open_workingtree()
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('lossless push between %r and %r not supported' % (self.branch_format_from, self.branch_format_to))
    m1 = mine.commit('my change')
    try:
        result = parent.branch.pull(mine.branch)
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('lossless push between %r and %r not supported' % (self.branch_format_from, self.branch_format_to))
    self.assertIsNot(None, result)
    self.assertIs(mine.branch, result.source_branch)
    self.assertIs(parent.branch, result.target_branch)
    self.assertIs(parent.branch, result.master_branch)
    self.assertIs(None, result.local_branch)
    self.assertEqual(1, result.old_revno)
    self.assertEqual(p1, result.old_revid)
    self.assertEqual(2, result.new_revno)
    self.assertEqual(m1, result.new_revid)
    self.assertEqual([], list(result.tag_conflicts))