from breezy import tests
from breezy.revision import NULL_REVISION
from breezy.tests import per_workingtree
def test_pull_overwrites(self):
    tree_a, tree_b, rev_a = self.get_pullable_trees()
    rev_b = tree_b.commit('foo')
    self.assertEqual(rev_b, tree_b.branch.last_revision())
    tree_b.pull(tree_a.branch, overwrite=True)
    self.assertTrue(tree_b.branch.repository.has_revision(rev_a))
    self.assertTrue(tree_b.branch.repository.has_revision(rev_b))
    self.assertEqual([rev_a], tree_b.get_parent_ids())