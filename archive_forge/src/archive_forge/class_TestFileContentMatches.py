from breezy.tests.per_intertree import TestCaseWithTwoTrees
class TestFileContentMatches(TestCaseWithTwoTrees):

    def test_same_contents_and_verifier(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        self.build_tree_contents([('1/file', b'apples'), ('2/file', b'apples')])
        tree1.add('file', ids=b'file-id-1')
        tree2.add('file', ids=b'file-id-2')
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        inter = self.intertree_class(tree1, tree2)
        self.assertTrue(inter.file_content_matches('file', 'file'))

    def test_different_contents_and_same_verifier(self):
        tree1 = self.make_branch_and_tree('1')
        tree2 = self.make_to_branch_and_tree('2')
        self.build_tree_contents([('1/file', b'apples'), ('2/file', b'oranges')])
        tree1.add('file', ids=b'file-id-1')
        tree2.add('file', ids=b'file-id-2')
        tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
        inter = self.intertree_class(tree1, tree2)
        self.assertFalse(inter.file_content_matches('file', 'file'))