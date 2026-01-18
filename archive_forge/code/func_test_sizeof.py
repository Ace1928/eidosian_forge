import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_sizeof(self):
    di = self._gc_module.DeltaIndex()
    lower_bound = di._max_num_sources * 12
    self.assertGreater(sys.getsizeof(di), lower_bound)