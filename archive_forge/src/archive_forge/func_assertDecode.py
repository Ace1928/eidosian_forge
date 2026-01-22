import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def assertDecode(self, exp_offset, exp_length, exp_newpos, data, pos):
    cmd = data[pos]
    pos += 1
    out = _groupcompress_py.decode_copy_instruction(data, cmd, pos)
    self.assertEqual((exp_offset, exp_length, exp_newpos), out)