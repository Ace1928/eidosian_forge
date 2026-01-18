import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_validate_correct_dirstate(self):
    state = self.create_complex_dirstate()
    state._validate()
    state.unlock()
    state.lock_read()
    try:
        state._validate()
    finally:
        state.unlock()