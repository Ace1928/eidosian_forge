import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_update_dir_to_file(self):
    """Directory becoming a file updates the entry."""
    state, entry = self.get_state_with_a()
    state.adjust_time(+10)
    self.create_and_test_dir(state, entry)
    os.rmdir('a')
    self.create_and_test_file(state, entry)