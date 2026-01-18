from breezy import errors, tests
from breezy.tests.per_tree import TestCaseWithTree
def test_path2id_nested_tree(self):
    tree, subtree = self.create_nested()
    self.assertIsNot(None, tree.path2id('subtree'))
    self.assertIsNot(None, tree.path2id('subtree/a'))
    self.assertEqual('subtree', tree.id2path(tree.path2id('subtree')))
    self.assertEqual('subtree/a', tree.id2path(tree.path2id('subtree/a')))
    self.assertIsNot('subtree/a', tree.id2path(tree.path2id('subtree/a'), recurse='down'))
    self.assertRaises(errors.NoSuchId, tree.id2path, tree.path2id('subtree/a'), recurse='none')