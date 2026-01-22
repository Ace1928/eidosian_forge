import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def assertEqualEncode(self, bytes, val):
    self.assertEqual(bytes, self._gc_module.encode_base128_int(val))