import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_second_add_source_triggers_make_index(self):
    di = self._gc_module.DeltaIndex()
    self.assertFalse(di._has_index())
    di.add_source(_text1, 0)
    self.assertFalse(di._has_index())
    di.add_source(_text2, 0)
    self.assertTrue(di._has_index())