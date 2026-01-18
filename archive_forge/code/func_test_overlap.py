import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_overlap(self):
    index = self.make_100_node_index()
    self.assertExpandOffsets([10, 11, 12, 13, 14, 15], index, [12, 13])
    self.assertExpandOffsets([10, 11, 12, 13, 14, 15], index, [11, 14])