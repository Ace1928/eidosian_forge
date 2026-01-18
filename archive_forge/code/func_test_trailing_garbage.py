import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_trailing_garbage(self):
    tree, state, expected = self.create_basic_dirstate()
    state.unlock()
    f = open('dirstate', 'ab')
    try:
        f.write(b'bogus\n')
    finally:
        f.close()
        state.lock_read()
    e = self.assertRaises(dirstate.DirstateCorrupt, state._read_dirblocks_if_needed)
    self.assertContainsRe(str(e), 'bogus')