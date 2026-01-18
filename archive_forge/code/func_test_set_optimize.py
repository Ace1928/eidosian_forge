import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_set_optimize(self):
    builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
    builder.set_optimize(for_size=True)
    self.assertTrue(builder._optimize_for_size)
    builder.set_optimize(for_size=False)
    self.assertFalse(builder._optimize_for_size)
    obj = object()
    builder._optimize_for_size = obj
    builder.set_optimize(combine_backing_indices=False)
    self.assertFalse(builder._combine_backing_indices)
    self.assertIs(obj, builder._optimize_for_size)
    builder.set_optimize(combine_backing_indices=True)
    self.assertTrue(builder._combine_backing_indices)
    self.assertIs(obj, builder._optimize_for_size)