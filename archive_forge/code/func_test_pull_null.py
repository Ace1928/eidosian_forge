from breezy import tests
from breezy.revision import NULL_REVISION
from breezy.tests import per_workingtree
def test_pull_null(self):
    tree_a, tree_b, rev_a = self.get_pullable_trees()
    if tree_a.supports_setting_file_ids():
        root_id = tree_a.path2id('')
        tree_a.pull(tree_b.branch, stop_revision=NULL_REVISION, overwrite=True)
        self.assertEqual(root_id, tree_a.path2id(''))
    else:
        tree_a.pull(tree_b.branch, stop_revision=NULL_REVISION, overwrite=True)