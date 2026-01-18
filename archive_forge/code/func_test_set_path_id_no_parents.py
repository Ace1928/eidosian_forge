import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_set_path_id_no_parents(self):
    """The id of a path can be changed trivally with no parents."""
    state = dirstate.DirState.initialize('dirstate')
    try:
        root_entry = ((b'', b'', b'TREE_ROOT'), [(b'd', b'', 0, False, b'x' * 32)])
        self.assertEqual([root_entry], list(state._iter_entries()))
        self.assertEqual(root_entry, state._get_entry(0, path_utf8=b''))
        self.assertEqual(root_entry, state._get_entry(0, fileid_utf8=b'TREE_ROOT'))
        self.assertEqual((None, None), state._get_entry(0, fileid_utf8=b'second-root-id'))
        state.set_path_id(b'', b'second-root-id')
        new_root_entry = ((b'', b'', b'second-root-id'), [(b'd', b'', 0, False, b'x' * 32)])
        expected_rows = [new_root_entry]
        self.assertEqual(expected_rows, list(state._iter_entries()))
        self.assertEqual(new_root_entry, state._get_entry(0, path_utf8=b''))
        self.assertEqual(new_root_entry, state._get_entry(0, fileid_utf8=b'second-root-id'))
        self.assertEqual((None, None), state._get_entry(0, fileid_utf8=b'TREE_ROOT'))
        state.save()
    finally:
        state.unlock()
    state = dirstate.DirState.on_file('dirstate')
    state.lock_read()
    try:
        state._validate()
        self.assertEqual(expected_rows, list(state._iter_entries()))
    finally:
        state.unlock()