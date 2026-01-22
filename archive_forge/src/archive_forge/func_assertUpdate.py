import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def assertUpdate(self, active, basis, target):
    """Assert that update_basis_by_delta works how we want.

        Set up a DirState object with active_shape for tree 0, basis_shape for
        tree 1. Then apply the delta from basis_shape to target_shape,
        and assert that the DirState is still valid, and that its stored
        content matches the target_shape.
        """
    active_tree = self.create_tree_from_shape(b'active', active)
    basis_tree = self.create_tree_from_shape(b'basis', basis)
    target_tree = self.create_tree_from_shape(b'target', target)
    state = self.create_empty_dirstate()
    state.set_state_from_scratch(active_tree.root_inventory, [(b'basis', basis_tree)], [])
    delta = target_tree.root_inventory._make_delta(basis_tree.root_inventory)
    state.update_basis_by_delta(delta, b'target')
    state._validate()
    dirstate_tree = workingtree_4.DirStateRevisionTree(state, b'target', _Repo(), None)
    self.assertEqual([], list(dirstate_tree.iter_changes(target_tree)))
    state2 = self.create_empty_dirstate()
    state2.set_state_from_scratch(active_tree.root_inventory, [(b'target', target_tree)], [])
    self.assertEqual(state2._dirblocks, state._dirblocks)
    return state