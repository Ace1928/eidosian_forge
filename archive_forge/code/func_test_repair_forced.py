from breezy import workingtree
from breezy.tests import TestCaseWithTransport
def test_repair_forced(self):
    tree = self.make_initial_tree()
    tree.rename_one('dir', 'alt_dir')
    self.assertTrue(tree.is_versioned('alt_dir'))
    self.run_bzr('repair-workingtree -d tree --force')
    self.assertFalse(tree.is_versioned('alt_dir'))
    self.assertPathExists('tree/alt_dir')