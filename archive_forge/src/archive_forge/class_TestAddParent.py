import os
from io import BytesIO
from ... import errors
from ... import revision as _mod_revision
from ...bzr.inventory import (Inventory, InventoryDirectory, InventoryFile,
from ...bzr.inventorytree import InventoryRevisionTree, InventoryTree
from ...tests import TestNotApplicable
from ...uncommit import uncommit
from .. import features
from ..per_workingtree import TestCaseWithWorkingTree
class TestAddParent(TestParents):

    def test_add_first_parent_id(self):
        """Test adding the first parent id"""
        tree = self.make_branch_and_tree('.')
        first_revision = tree.commit('first post')
        uncommit(tree.branch, tree=tree)
        tree.add_parent_tree_id(first_revision)
        self.assertConsistentParents([first_revision], tree)

    def test_add_first_parent_id_ghost_rejects(self):
        """Test adding the first parent id - as a ghost"""
        tree = self.make_branch_and_tree('.')
        self.assertRaises(errors.GhostRevisionUnusableHere, tree.add_parent_tree_id, b'first-revision')

    def test_add_first_parent_id_ghost_force(self):
        """Test adding the first parent id - as a ghost"""
        tree = self.make_branch_and_tree('.')
        try:
            tree.add_parent_tree_id(b'first-revision', allow_leftmost_as_ghost=True)
        except errors.GhostRevisionUnusableHere:
            self.assertFalse(tree._format.supports_leftmost_parent_id_as_ghost)
        else:
            self.assertTrue(tree._format.supports_leftmost_parent_id_as_ghost)
            self.assertConsistentParents([b'first-revision'], tree)

    def test_add_second_parent_id_with_ghost_first(self):
        """Test adding the second parent when the first is a ghost."""
        tree = self.make_branch_and_tree('.')
        try:
            tree.add_parent_tree_id(b'first-revision', allow_leftmost_as_ghost=True)
        except errors.GhostRevisionUnusableHere:
            self.assertFalse(tree._format.supports_leftmost_parent_id_as_ghost)
        else:
            tree.add_parent_tree_id(b'second')
            self.assertConsistentParents([b'first-revision', b'second'], tree)

    def test_add_second_parent_id(self):
        """Test adding the second parent id"""
        tree = self.make_branch_and_tree('.')
        first_revision = tree.commit('first post')
        uncommit(tree.branch, tree=tree)
        second_revision = tree.commit('second post')
        tree.add_parent_tree_id(first_revision)
        self.assertConsistentParents([second_revision, first_revision], tree)

    def test_add_second_parent_id_ghost(self):
        """Test adding the second parent id - as a ghost"""
        tree = self.make_branch_and_tree('.')
        first_revision = tree.commit('first post')
        if tree._format.supports_righthand_parent_id_as_ghost:
            tree.add_parent_tree_id(b'second')
            self.assertConsistentParents([first_revision, b'second'], tree)
        else:
            self.assertRaises(errors.GhostRevisionUnusableHere, tree.add_parent_tree_id, b'second')

    def test_add_first_parent_tree(self):
        """Test adding the first parent id"""
        tree = self.make_branch_and_tree('.')
        first_revision = tree.commit('first post')
        uncommit(tree.branch, tree=tree)
        tree.add_parent_tree((first_revision, tree.branch.repository.revision_tree(first_revision)))
        self.assertConsistentParents([first_revision], tree)

    def test_add_first_parent_tree_ghost_rejects(self):
        """Test adding the first parent id - as a ghost"""
        tree = self.make_branch_and_tree('.')
        self.assertRaises(errors.GhostRevisionUnusableHere, tree.add_parent_tree, (b'first-revision', None))

    def test_add_first_parent_tree_ghost_force(self):
        """Test adding the first parent id - as a ghost"""
        tree = self.make_branch_and_tree('.')
        try:
            tree.add_parent_tree((b'first-revision', None), allow_leftmost_as_ghost=True)
        except errors.GhostRevisionUnusableHere:
            self.assertFalse(tree._format.supports_leftmost_parent_id_as_ghost)
        else:
            self.assertTrue(tree._format.supports_leftmost_parent_id_as_ghost)
            self.assertConsistentParents([b'first-revision'], tree)

    def test_add_second_parent_tree(self):
        """Test adding the second parent id"""
        tree = self.make_branch_and_tree('.')
        first_revision = tree.commit('first post')
        uncommit(tree.branch, tree=tree)
        second_revision = tree.commit('second post')
        tree.add_parent_tree((first_revision, tree.branch.repository.revision_tree(first_revision)))
        self.assertConsistentParents([second_revision, first_revision], tree)

    def test_add_second_parent_tree_ghost(self):
        """Test adding the second parent id - as a ghost"""
        tree = self.make_branch_and_tree('.')
        first_revision = tree.commit('first post')
        if tree._format.supports_righthand_parent_id_as_ghost:
            tree.add_parent_tree((b'second', None))
            self.assertConsistentParents([first_revision, b'second'], tree)
        else:
            self.assertRaises(errors.GhostRevisionUnusableHere, tree.add_parent_tree, (b'second', None))