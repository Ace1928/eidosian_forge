from breezy import errors, tests
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_reset_state_forgets_changes(self):
    tree = self.make_initial_tree()
    tree.rename_one('foo', 'baz')
    self.assertFalse(tree.is_versioned('foo'))
    if tree.supports_rename_tracking() and tree.supports_file_ids:
        foo_id = tree.basis_tree().path2id('foo')
        self.assertEqual(foo_id, tree.path2id('baz'))
    else:
        self.assertTrue(tree.is_versioned('baz'))
    tree.reset_state()
    if tree.supports_file_ids:
        self.assertEqual(foo_id, tree.path2id('foo'))
        self.assertEqual(None, tree.path2id('baz'))
    self.assertPathDoesNotExist('tree/foo')
    self.assertPathExists('tree/baz')