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
class TestSetParents(TestParents):

    def test_set_no_parents(self):
        t = self.make_branch_and_tree('.')
        t.set_parent_trees([])
        self.assertEqual([], t.get_parent_ids())
        t.commit('first post')
        t.set_parent_trees([])
        self.assertConsistentParents([], t)

    def test_set_null_parent(self):
        t = self.make_branch_and_tree('.')
        self.assertRaises(errors.ReservedId, t.set_parent_ids, [b'null:'], allow_leftmost_as_ghost=True)
        self.assertRaises(errors.ReservedId, t.set_parent_trees, [(b'null:', None)], allow_leftmost_as_ghost=True)

    def test_set_one_ghost_parent_rejects(self):
        t = self.make_branch_and_tree('.')
        self.assertRaises(errors.GhostRevisionUnusableHere, t.set_parent_trees, [(b'missing-revision-id', None)])

    def test_set_one_ghost_parent_force(self):
        t = self.make_branch_and_tree('.')
        if t._format.supports_leftmost_parent_id_as_ghost:
            t.set_parent_trees([(b'missing-revision-id', None)], allow_leftmost_as_ghost=True)
            self.assertConsistentParents([b'missing-revision-id'], t)
        else:
            self.assertRaises(errors.GhostRevisionUnusableHere, t.set_parent_trees, [(b'missing-revision-id', None)])
            self.assertConsistentParents([], t)

    def test_set_two_parents_one_ghost(self):
        t = self.make_branch_and_tree('.')
        revision_in_repo = t.commit('first post')
        uncommit(t.branch, tree=t)
        rev_tree = t.branch.repository.revision_tree(revision_in_repo)
        if t._format.supports_righthand_parent_id_as_ghost:
            t.set_parent_trees([(revision_in_repo, rev_tree), (b'another-missing', None)])
            self.assertConsistentParents([revision_in_repo, b'another-missing'], t)
        else:
            self.assertRaises(errors.GhostRevisionUnusableHere, t.set_parent_trees, [(revision_in_repo, rev_tree), (b'another-missing', None)])

    def test_set_three_parents(self):
        t = self.make_branch_and_tree('.')
        first_revision = t.commit('first post')
        uncommit(t.branch, tree=t)
        second_revision = t.commit('second post')
        uncommit(t.branch, tree=t)
        third_revision = t.commit('third post')
        uncommit(t.branch, tree=t)
        rev_tree1 = t.branch.repository.revision_tree(first_revision)
        rev_tree2 = t.branch.repository.revision_tree(second_revision)
        rev_tree3 = t.branch.repository.revision_tree(third_revision)
        t.set_parent_trees([(first_revision, rev_tree1), (second_revision, rev_tree2), (third_revision, rev_tree3)])
        self.assertConsistentParents([first_revision, second_revision, third_revision], t)

    def test_set_no_parents_ids(self):
        t = self.make_branch_and_tree('.')
        t.set_parent_ids([])
        self.assertEqual([], t.get_parent_ids())
        t.commit('first post')
        t.set_parent_ids([])
        self.assertConsistentParents([], t)

    def test_set_one_ghost_parent_ids_rejects(self):
        t = self.make_branch_and_tree('.')
        self.assertRaises(errors.GhostRevisionUnusableHere, t.set_parent_ids, [b'missing-revision-id'])

    def test_set_one_ghost_parent_ids_force(self):
        t = self.make_branch_and_tree('.')
        if t._format.supports_leftmost_parent_id_as_ghost:
            t.set_parent_ids([b'missing-revision-id'], allow_leftmost_as_ghost=True)
            self.assertConsistentParents([b'missing-revision-id'], t)
        else:
            self.assertRaises(errors.GhostRevisionUnusableHere, t.set_parent_ids, [b'missing-revision-id'], allow_leftmost_as_ghost=True)

    def test_set_two_parents_one_ghost_ids(self):
        t = self.make_branch_and_tree('.')
        revision_in_repo = t.commit('first post')
        uncommit(t.branch, tree=t)
        rev_tree = t.branch.repository.revision_tree(revision_in_repo)
        if t._format.supports_righthand_parent_id_as_ghost:
            t.set_parent_ids([revision_in_repo, b'another-missing'])
            self.assertConsistentParents([revision_in_repo, b'another-missing'], t)
        else:
            self.assertRaises(errors.GhostRevisionUnusableHere, t.set_parent_ids, [revision_in_repo, b'another-missing'])

    def test_set_three_parents_ids(self):
        t = self.make_branch_and_tree('.')
        first_revision = t.commit('first post')
        uncommit(t.branch, tree=t)
        second_revision = t.commit('second post')
        uncommit(t.branch, tree=t)
        third_revision = t.commit('third post')
        uncommit(t.branch, tree=t)
        rev_tree1 = t.branch.repository.revision_tree(first_revision)
        rev_tree2 = t.branch.repository.revision_tree(second_revision)
        rev_tree3 = t.branch.repository.revision_tree(third_revision)
        t.set_parent_ids([first_revision, second_revision, third_revision])
        self.assertConsistentParents([first_revision, second_revision, third_revision], t)

    def test_set_duplicate_parent_ids(self):
        t = self.make_branch_and_tree('.')
        rev1 = t.commit('first post')
        uncommit(t.branch, tree=t)
        rev2 = t.commit('second post')
        uncommit(t.branch, tree=t)
        rev3 = t.commit('third post')
        uncommit(t.branch, tree=t)
        t.set_parent_ids([rev1, rev2, rev2, rev3])
        self.assertConsistentParents([rev1, rev2, rev3], t)

    def test_set_duplicate_parent_trees(self):
        t = self.make_branch_and_tree('.')
        rev1 = t.commit('first post')
        uncommit(t.branch, tree=t)
        rev2 = t.commit('second post')
        uncommit(t.branch, tree=t)
        rev3 = t.commit('third post')
        uncommit(t.branch, tree=t)
        rev_tree1 = t.branch.repository.revision_tree(rev1)
        rev_tree2 = t.branch.repository.revision_tree(rev2)
        rev_tree3 = t.branch.repository.revision_tree(rev3)
        t.set_parent_trees([(rev1, rev_tree1), (rev2, rev_tree2), (rev2, rev_tree2), (rev3, rev_tree3)])
        self.assertConsistentParents([rev1, rev2, rev3], t)

    def test_set_parent_ids_in_ancestry(self):
        t = self.make_branch_and_tree('.')
        rev1 = t.commit('first post')
        rev2 = t.commit('second post')
        rev3 = t.commit('third post')
        t.set_parent_ids([rev1])
        t.branch.set_last_revision_info(1, rev1)
        self.assertConsistentParents([rev1], t)
        t.set_parent_ids([rev1, rev2, rev3])
        self.assertConsistentParents([rev1, rev3], t)
        t.set_parent_ids([rev2, rev3, rev1])
        self.assertConsistentParents([rev2, rev3], t)

    def test_set_parent_trees_in_ancestry(self):
        t = self.make_branch_and_tree('.')
        rev1 = t.commit('first post')
        rev2 = t.commit('second post')
        rev3 = t.commit('third post')
        t.set_parent_ids([rev1])
        t.branch.set_last_revision_info(1, rev1)
        self.assertConsistentParents([rev1], t)
        rev_tree1 = t.branch.repository.revision_tree(rev1)
        rev_tree2 = t.branch.repository.revision_tree(rev2)
        rev_tree3 = t.branch.repository.revision_tree(rev3)
        t.set_parent_trees([(rev1, rev_tree1), (rev2, rev_tree2), (rev3, rev_tree3)])
        self.assertConsistentParents([rev1, rev3], t)
        t.set_parent_trees([(rev2, rev_tree2), (rev1, rev_tree1), (rev3, rev_tree3)])
        self.assertConsistentParents([rev2, rev3], t)

    def test_unicode_symlink(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('tree1')
        target = 'Ω'
        link_name = '€link'
        os.symlink(target, 'tree1/' + link_name)
        tree.add([link_name])
        revision1 = tree.commit('added a link to a Unicode target')
        revision2 = tree.commit('this revision will be discarded')
        tree.set_parent_ids([revision1])
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual(target, tree.get_symlink_target(link_name))
        basis = tree.basis_tree()
        self.assertEqual(target, basis.get_symlink_target(link_name))