from breezy import branch, controldir, errors, revision
from breezy.tests import TestNotApplicable, fixtures, per_branch
def test_pull_overwrite_set(self):
    tree_a = self.make_branch_and_tree('tree_a')
    rev1 = tree_a.commit('message 1')
    tree_b = tree_a.controldir.sprout('tree_b').open_workingtree()
    rev2a = tree_a.commit('message 2a')
    rev2b = tree_b.commit('message 2b')
    self.assertRaises(errors.DivergedBranches, tree_a.pull, tree_b.branch)
    self.assertRaises(errors.DivergedBranches, tree_a.branch.pull, tree_b.branch, overwrite=set(), stop_revision=rev2b)
    self.assertEqual(rev2a, tree_a.branch.last_revision())
    if tree_a.branch.repository._format.supports_unreferenced_revisions:
        self.assertTrue(tree_a.branch.repository.has_revision(rev2b))
    tree_a.branch.pull(tree_b.branch, overwrite={'history'}, stop_revision=rev2b)
    self.assertEqual(rev2b, tree_a.branch.last_revision())
    self.assertEqual(tree_b.branch.last_revision(), tree_a.branch.last_revision())
    tree_a.branch.pull(tree_b.branch, overwrite={'history', 'tags'}, stop_revision=rev2b)