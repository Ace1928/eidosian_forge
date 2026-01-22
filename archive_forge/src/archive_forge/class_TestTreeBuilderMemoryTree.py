from breezy import tests
from breezy.memorytree import MemoryTree
from breezy.tests import TestCaseWithTransport
from breezy.treebuilder import AlreadyBuilding, NotBuilding, TreeBuilder
class TestTreeBuilderMemoryTree(tests.TestCaseWithMemoryTransport):

    def test_create(self):
        TreeBuilder()

    def test_start_tree_locks_write(self):
        builder = TreeBuilder()
        tree = FakeTree()
        builder.start_tree(tree)
        self.assertEqual(['lock_tree_write'], tree._calls)

    def test_start_tree_when_started_fails(self):
        builder = TreeBuilder()
        tree = FakeTree()
        builder.start_tree(tree)
        self.assertRaises(AlreadyBuilding, builder.start_tree, tree)

    def test_finish_tree_not_started_errors(self):
        builder = TreeBuilder()
        self.assertRaises(NotBuilding, builder.finish_tree)

    def test_finish_tree_unlocks(self):
        builder = TreeBuilder()
        tree = FakeTree()
        builder.start_tree(tree)
        builder.finish_tree()
        self.assertEqual(['lock_tree_write', 'unlock'], tree._calls)

    def test_build_tree_not_started_errors(self):
        builder = TreeBuilder()
        self.assertRaises(NotBuilding, builder.build, 'foo')

    def test_build_tree(self):
        """Test building works using a MemoryTree."""
        branch = self.make_branch('branch')
        tree = MemoryTree.create_on_branch(branch)
        builder = TreeBuilder()
        builder.start_tree(tree)
        builder.build(['foo', 'bar/', 'bar/file'])
        self.assertEqual(b'contents of foo\n', tree.get_file('foo').read())
        self.assertEqual(b'contents of bar/file\n', tree.get_file('bar/file').read())
        builder.finish_tree()