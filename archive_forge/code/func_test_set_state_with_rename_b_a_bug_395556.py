import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_set_state_with_rename_b_a_bug_395556(self):
    tree1 = self.make_branch_and_tree('tree1')
    self.build_tree(['tree1/b'])
    with tree1.lock_write():
        tree1.add(['b'], ids=[b'b-id'])
        root_id = tree1.path2id('')
        inv = tree1.root_inventory
        state = dirstate.DirState.initialize('dirstate')
        try:
            state.set_state_from_inventory(inv)
            inv.rename(b'b-id', root_id, 'a')
            state.set_state_from_inventory(inv)
            expected_result1 = [(b'', b'', root_id, b'd'), (b'', b'a', b'b-id', b'f')]
            values = []
            for entry in state._iter_entries():
                values.append(entry[0] + entry[1][0][:1])
            self.assertEqual(expected_result1, values)
        finally:
            state.unlock()