from breezy import tests
from breezy.memorytree import MemoryTree
from breezy.tests import TestCaseWithTransport
from breezy.treebuilder import AlreadyBuilding, NotBuilding, TreeBuilder
def test_start_tree_locks_write(self):
    builder = TreeBuilder()
    tree = FakeTree()
    builder.start_tree(tree)
    self.assertEqual(['lock_tree_write'], tree._calls)