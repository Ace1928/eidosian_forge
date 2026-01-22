import stat
from dulwich.objects import Blob, Tree
from breezy.bzr.inventorytree import InventoryTreeChange as TreeChange
from breezy.delta import TreeDelta
from breezy.errors import PathsNotVersionedError
from breezy.git.mapping import default_mapping
from breezy.git.tree import (changes_from_git_changes,
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.workingtree import WorkingTree
class FindRelatedPathsAcrossTrees(TestCaseWithTransport):

    def test_none(self):
        self.make_branch_and_tree('t1', format='git')
        wt = WorkingTree.open('t1')
        self.assertIs(None, wt.find_related_paths_across_trees(None))

    def test_empty(self):
        self.make_branch_and_tree('t1', format='git')
        wt = WorkingTree.open('t1')
        self.assertEqual([], list(wt.find_related_paths_across_trees([])))

    def test_directory(self):
        self.make_branch_and_tree('t1', format='git')
        wt = WorkingTree.open('t1')
        self.build_tree(['t1/dir/', 't1/dir/file'])
        wt.add(['dir', 'dir/file'])
        self.assertEqual(['dir/file'], list(wt.find_related_paths_across_trees(['dir/file'])))
        self.assertEqual(['dir'], list(wt.find_related_paths_across_trees(['dir'])))

    def test_empty_directory(self):
        self.make_branch_and_tree('t1', format='git')
        wt = WorkingTree.open('t1')
        self.build_tree(['t1/dir/'])
        wt.add(['dir'])
        self.assertEqual(['dir'], list(wt.find_related_paths_across_trees(['dir'])))
        self.assertRaises(PathsNotVersionedError, wt.find_related_paths_across_trees, ['dir/file'])

    def test_missing(self):
        self.make_branch_and_tree('t1', format='git')
        wt = WorkingTree.open('t1')
        self.assertRaises(PathsNotVersionedError, wt.find_related_paths_across_trees, ['file'])

    def test_not_versioned(self):
        self.make_branch_and_tree('t1', format='git')
        self.make_branch_and_tree('t2', format='git')
        wt1 = WorkingTree.open('t1')
        wt2 = WorkingTree.open('t2')
        self.build_tree(['t1/file'])
        self.build_tree(['t2/file'])
        self.assertRaises(PathsNotVersionedError, wt1.find_related_paths_across_trees, ['file'], [wt2])

    def test_single(self):
        self.make_branch_and_tree('t1', format='git')
        wt = WorkingTree.open('t1')
        self.build_tree(['t1/file'])
        wt.add('file')
        self.assertEqual(['file'], list(wt.find_related_paths_across_trees(['file'])))