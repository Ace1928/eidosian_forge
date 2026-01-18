import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_read_all_from_root(self):
    index = self.make_index(4096 * 10, 20)
    self.assertExpandOffsets(list(range(10)), index, [0])