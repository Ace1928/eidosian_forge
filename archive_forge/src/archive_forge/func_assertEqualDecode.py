import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def assertEqualDecode(self, val, num_decode, bytes):
    self.assertEqual((val, num_decode), self._gc_module.decode_base128_int(bytes))