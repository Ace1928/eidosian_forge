import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_set_state_from_inventory_mixed_paths(self):
    tree1 = self.make_branch_and_tree('tree1')
    self.build_tree(['tree1/a/', 'tree1/a/b/', 'tree1/a-b/', 'tree1/a/b/foo', 'tree1/a-b/bar'])
    tree1.lock_write()
    try:
        tree1.add(['a', 'a/b', 'a-b', 'a/b/foo', 'a-b/bar'], ids=[b'a-id', b'b-id', b'a-b-id', b'foo-id', b'bar-id'])
        tree1.commit('rev1', rev_id=b'rev1')
        root_id = tree1.path2id('')
        inv = tree1.root_inventory
    finally:
        tree1.unlock()
    expected_result1 = [(b'', b'', root_id, b'd'), (b'', b'a', b'a-id', b'd'), (b'', b'a-b', b'a-b-id', b'd'), (b'a', b'b', b'b-id', b'd'), (b'a/b', b'foo', b'foo-id', b'f'), (b'a-b', b'bar', b'bar-id', b'f')]
    expected_result2 = [(b'', b'', root_id, b'd'), (b'', b'a', b'a-id', b'd'), (b'', b'a-b', b'a-b-id', b'd'), (b'a-b', b'bar', b'bar-id', b'f')]
    state = dirstate.DirState.initialize('dirstate')
    try:
        state.set_state_from_inventory(inv)
        values = []
        for entry in state._iter_entries():
            values.append(entry[0] + entry[1][0][:1])
        self.assertEqual(expected_result1, values)
        inv.delete(b'b-id')
        state.set_state_from_inventory(inv)
        values = []
        for entry in state._iter_entries():
            values.append(entry[0] + entry[1][0][:1])
        self.assertEqual(expected_result2, values)
    finally:
        state.unlock()