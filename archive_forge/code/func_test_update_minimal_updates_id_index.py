import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_update_minimal_updates_id_index(self):
    state = self.create_dirstate_with_root_and_subdir()
    self.addCleanup(state.unlock)
    id_index = state._get_id_index()
    self.assertEqual([b'a-root-value', b'subdir-id'], sorted(id_index))
    state.add('file-name', b'file-id', 'file', None, '')
    self.assertEqual([b'a-root-value', b'file-id', b'subdir-id'], sorted(id_index))
    state.update_minimal((b'', b'new-name', b'file-id'), b'f', path_utf8=b'new-name')
    self.assertEqual([b'a-root-value', b'file-id', b'subdir-id'], sorted(id_index))
    self.assertEqual([(b'', b'new-name', b'file-id')], sorted(id_index[b'file-id']))
    state._validate()