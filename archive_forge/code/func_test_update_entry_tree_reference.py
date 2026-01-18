import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_update_entry_tree_reference(self):
    state = test_dirstate.InstrumentedDirState.initialize('dirstate')
    self.addCleanup(state.unlock)
    state.add('r', b'r-id', 'tree-reference', None, b'')
    self.build_tree(['r/'])
    entry = state._get_entry(0, path_utf8=b'r')
    self.do_update_entry(state, entry, 'r')
    entry = state._get_entry(0, path_utf8=b'r')
    self.assertEqual(b't', entry[1][0][0])