import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
class TestMemRChr(tests.TestCase):
    """Test memrchr functionality"""
    _test_needs_features = [compiled_dirstate_helpers_feature]

    def assertMemRChr(self, expected, s, c):
        from .._dirstate_helpers_pyx import _py_memrchr
        self.assertEqual(expected, _py_memrchr(s, c))

    def test_missing(self):
        self.assertMemRChr(None, b'', b'a')
        self.assertMemRChr(None, b'', b'c')
        self.assertMemRChr(None, b'abcdefghijklm', b'q')
        self.assertMemRChr(None, b'aaaaaaaaaaaaaaaaaaaaaaa', b'b')

    def test_single_entry(self):
        self.assertMemRChr(0, b'abcdefghijklm', b'a')
        self.assertMemRChr(1, b'abcdefghijklm', b'b')
        self.assertMemRChr(2, b'abcdefghijklm', b'c')
        self.assertMemRChr(10, b'abcdefghijklm', b'k')
        self.assertMemRChr(11, b'abcdefghijklm', b'l')
        self.assertMemRChr(12, b'abcdefghijklm', b'm')

    def test_multiple(self):
        self.assertMemRChr(10, b'abcdefjklmabcdefghijklm', b'a')
        self.assertMemRChr(11, b'abcdefjklmabcdefghijklm', b'b')
        self.assertMemRChr(12, b'abcdefjklmabcdefghijklm', b'c')
        self.assertMemRChr(20, b'abcdefjklmabcdefghijklm', b'k')
        self.assertMemRChr(21, b'abcdefjklmabcdefghijklm', b'l')
        self.assertMemRChr(22, b'abcdefjklmabcdefghijklm', b'm')
        self.assertMemRChr(22, b'aaaaaaaaaaaaaaaaaaaaaaa', b'a')

    def test_with_nulls(self):
        self.assertMemRChr(10, b'abc\x00\x00\x00jklmabc\x00\x00\x00ghijklm', b'a')
        self.assertMemRChr(11, b'abc\x00\x00\x00jklmabc\x00\x00\x00ghijklm', b'b')
        self.assertMemRChr(12, b'abc\x00\x00\x00jklmabc\x00\x00\x00ghijklm', b'c')
        self.assertMemRChr(20, b'abc\x00\x00\x00jklmabc\x00\x00\x00ghijklm', b'k')
        self.assertMemRChr(21, b'abc\x00\x00\x00jklmabc\x00\x00\x00ghijklm', b'l')
        self.assertMemRChr(22, b'abc\x00\x00\x00jklmabc\x00\x00\x00ghijklm', b'm')
        self.assertMemRChr(22, b'aaa\x00\x00\x00aaaaaaa\x00\x00\x00aaaaaaa', b'a')
        self.assertMemRChr(9, b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', b'\x00')