import os
from breezy import osutils, tests, workingtree
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
class TestSmartAddTree(TestCaseWithWorkingTree):

    def setUp(self):
        super().setUp()
        self.requireFeature(features.SymlinkFeature(self.test_dir))

    def test_smart_add_symlink(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/link@', b'target')])
        tree.smart_add(['tree/link'])
        self.assertTrue(tree.is_versioned('link'))
        self.assertFalse(tree.is_versioned('target'))
        self.assertEqual('symlink', tree.kind('link'))

    def test_smart_add_symlink_pointing_outside(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/link@', '../../../../target')])
        tree.smart_add(['tree/link'])
        self.assertTrue(tree.is_versioned('link'))
        self.assertFalse(tree.is_versioned('target'))
        self.assertEqual('symlink', tree.kind('link'))

    def test_add_file_under_symlink(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/link@', 'dir'), ('tree/dir/',), ('tree/dir/file', b'content')])
        if tree.has_versioned_directories():
            self.assertEqual(tree.smart_add(['tree/link/file']), (['dir', 'dir/file'], {}))
        else:
            self.assertEqual(tree.smart_add(['tree/link/file']), (['dir/file'], {}))
        self.assertTrue(tree.is_versioned('dir/file'))
        self.assertTrue(tree.is_versioned('dir'))
        self.assertFalse(tree.is_versioned('link'))
        self.assertFalse(tree.is_versioned('link/file'))