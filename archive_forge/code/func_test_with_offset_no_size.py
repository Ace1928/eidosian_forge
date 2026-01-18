import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_with_offset_no_size(self):
    index = self.make_index_with_offset(key_elements=1, ref_lists=1, offset=1234, nodes=self.make_nodes(200, 1, 1))
    index._size = None
    self.assertEqual(200, index.key_count())