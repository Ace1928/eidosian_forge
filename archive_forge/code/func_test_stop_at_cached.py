import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_stop_at_cached(self):
    index = self.make_100_node_index()
    self.set_cached_offsets(index, [0, 10, 19])
    self.assertExpandOffsets([11, 12, 13, 14, 15, 16], index, [11])
    self.assertExpandOffsets([11, 12, 13, 14, 15, 16], index, [12])
    self.assertExpandOffsets([12, 13, 14, 15, 16, 17, 18], index, [15])
    self.assertExpandOffsets([13, 14, 15, 16, 17, 18], index, [16])
    self.assertExpandOffsets([13, 14, 15, 16, 17, 18], index, [17])
    self.assertExpandOffsets([13, 14, 15, 16, 17, 18], index, [18])