from breezy import tests
from breezy.revision import NULL_REVISION
from breezy.tests import per_workingtree
def test_pull_merges_tree_content(self):
    tree_a, tree_b, rev_a = self.get_pullable_trees()
    tree_b.pull(tree_a.branch)
    self.assertFileEqual(b'contents of from/file\n', 'to/file')