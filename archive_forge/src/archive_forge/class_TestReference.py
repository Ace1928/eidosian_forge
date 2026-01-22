from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
class TestReference(TestCaseWithTree):

    def skip_if_no_reference(self, tree):
        if not tree.supports_tree_reference():
            raise tests.TestNotApplicable('Tree references not supported')

    def create_nested(self):
        work_tree = self.make_branch_and_tree('wt')
        with work_tree.lock_write():
            self.skip_if_no_reference(work_tree)
            subtree = self.make_branch_and_tree('wt/subtree')
            subtree.commit('foo')
            work_tree.add_reference(subtree)
        tree = self._convert_tree(work_tree)
        self.skip_if_no_reference(tree)
        return (tree, subtree)

    def test_get_reference_revision(self):
        tree, subtree = self.create_nested()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual(subtree.last_revision(), tree.get_reference_revision('subtree'))

    def test_iter_references(self):
        tree, subtree = self.create_nested()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual(['subtree'], list(tree.iter_references()))

    def test_get_nested_tree(self):
        tree, subtree = self.create_nested()
        try:
            changes = subtree.changes_from(tree.get_nested_tree('subtree'))
            self.assertFalse(changes.has_changed())
        except MissingNestedTree:
            pass

    def test_get_root_id(self):
        tree = self.make_branch_and_tree('tree')
        if not tree.supports_file_ids:
            raise tests.TestNotApplicable('file ids not supported')
        root_id = tree.path2id('')
        if root_id is not None:
            self.assertIsInstance(root_id, bytes)

    def test_is_versioned(self):
        tree = self.make_branch_and_tree('tree')
        self.assertTrue(tree.is_versioned(''))
        self.assertFalse(tree.is_versioned('blah'))
        self.build_tree(['tree/dir/', 'tree/dir/file'])
        self.assertFalse(tree.is_versioned('dir'))
        self.assertFalse(tree.is_versioned('dir/'))
        tree.add(['dir', 'dir/file'])
        self.assertTrue(tree.is_versioned('dir'))
        self.assertTrue(tree.is_versioned('dir/'))