import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_with_nulls(self):
    self.assertMemRChr(10, b'abc\x00\x00\x00jklmabc\x00\x00\x00ghijklm', b'a')
    self.assertMemRChr(11, b'abc\x00\x00\x00jklmabc\x00\x00\x00ghijklm', b'b')
    self.assertMemRChr(12, b'abc\x00\x00\x00jklmabc\x00\x00\x00ghijklm', b'c')
    self.assertMemRChr(20, b'abc\x00\x00\x00jklmabc\x00\x00\x00ghijklm', b'k')
    self.assertMemRChr(21, b'abc\x00\x00\x00jklmabc\x00\x00\x00ghijklm', b'l')
    self.assertMemRChr(22, b'abc\x00\x00\x00jklmabc\x00\x00\x00ghijklm', b'm')
    self.assertMemRChr(22, b'aaa\x00\x00\x00aaaaaaa\x00\x00\x00aaaaaaa', b'a')
    self.assertMemRChr(9, b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', b'\x00')