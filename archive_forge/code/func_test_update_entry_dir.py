import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_update_entry_dir(self):
    state, entry = self.get_state_with_a()
    self.build_tree(['a/'])
    self.assertIs(None, self.do_update_entry(state, entry, b'a'))