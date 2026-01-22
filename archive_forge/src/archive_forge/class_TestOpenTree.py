import os
from breezy import osutils, tests, workingtree
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
class TestOpenTree(TestCaseWithWorkingTree):

    def setUp(self):
        super().setUp()
        self.requireFeature(features.SymlinkFeature(self.test_dir))

    def test_open_containing_through_symlink(self):
        self.make_test_tree()
        self.check_open_containing('link/content', 'tree', 'content')
        self.check_open_containing('link/sublink', 'tree', 'sublink')
        self.check_open_containing('link/sublink/subcontent', 'tree', 'sublink/subcontent')

    def check_open_containing(self, to_open, expected_tree_name, expected_relpath):
        wt, relpath = workingtree.WorkingTree.open_containing(to_open)
        self.assertEqual(relpath, expected_relpath)
        self.assertEndsWith(wt.basedir, expected_tree_name)

    def test_tree_files(self):
        self.make_test_tree()
        self.check_tree_files(['tree/outerlink'], 'tree', ['outerlink'])
        self.check_tree_files(['link/outerlink'], 'tree', ['outerlink'])
        self.check_tree_files(['link/sublink/subcontent'], 'tree', ['subdir/subcontent'])

    def check_tree_files(self, to_open, expected_tree, expect_paths):
        tree, relpaths = workingtree.WorkingTree.open_containing_paths(to_open)
        self.assertEndsWith(tree.basedir, expected_tree)
        self.assertEqual(expect_paths, relpaths)

    def make_test_tree(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('link@', 'tree'), ('tree/outerlink@', '/not/there'), ('tree/content', b'hello'), ('tree/sublink@', 'subdir'), ('tree/subdir/',), ('tree/subdir/subcontent', b'subcontent stuff')])