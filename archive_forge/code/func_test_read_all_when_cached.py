import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_read_all_when_cached(self):
    index = self.make_index(4096 * 10, 5)
    self.prepare_index(index, node_ref_lists=0, key_length=1, key_count=1000, row_lengths=[1, 9], cached_offsets=[0, 1, 2, 5, 6])
    self.assertExpandOffsets([3, 4, 7, 8, 9], index, [3])
    self.assertExpandOffsets([3, 4, 7, 8, 9], index, [8])
    self.assertExpandOffsets([3, 4, 7, 8, 9], index, [9])