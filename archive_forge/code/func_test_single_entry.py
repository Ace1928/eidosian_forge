import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_single_entry(self):
    self.assertMemRChr(0, b'abcdefghijklm', b'a')
    self.assertMemRChr(1, b'abcdefghijklm', b'b')
    self.assertMemRChr(2, b'abcdefghijklm', b'c')
    self.assertMemRChr(10, b'abcdefghijklm', b'k')
    self.assertMemRChr(11, b'abcdefghijklm', b'l')
    self.assertMemRChr(12, b'abcdefghijklm', b'm')