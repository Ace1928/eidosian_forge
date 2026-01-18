from breezy import errors, transport
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unversion_subtree(self):
    """Unversioning the root of a subtree unversions the entire subtree."""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/b', 'c'])
    tree.add(['a', 'a/b', 'c'])
    with tree.lock_write():
        tree.unversion(['a'])
        self.assertFalse(tree.is_versioned('a'))
        self.assertFalse(tree.is_versioned('a/b'))
        self.assertTrue(tree.is_versioned('c'))
        self.assertTrue(tree.has_filename('a'))
        self.assertTrue(tree.has_filename('a/b'))
        self.assertTrue(tree.has_filename('c'))