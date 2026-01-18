import stat
from dulwich.objects import Blob, Tree
from breezy.bzr.inventorytree import InventoryTreeChange as TreeChange
from breezy.delta import TreeDelta
from breezy.errors import PathsNotVersionedError
from breezy.git.mapping import default_mapping
from breezy.git.tree import (changes_from_git_changes,
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_not_versioned(self):
    self.make_branch_and_tree('t1', format='git')
    self.make_branch_and_tree('t2', format='git')
    wt1 = WorkingTree.open('t1')
    wt2 = WorkingTree.open('t2')
    self.build_tree(['t1/file'])
    self.build_tree(['t2/file'])
    self.assertRaises(PathsNotVersionedError, wt1.find_related_paths_across_trees, ['file'], [wt2])