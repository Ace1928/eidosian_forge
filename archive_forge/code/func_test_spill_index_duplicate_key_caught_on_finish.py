import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_spill_index_duplicate_key_caught_on_finish(self):
    builder = btree_index.BTreeBuilder(key_elements=1, spill_at=2)
    nodes = [node[0:2] for node in self.make_nodes(16, 1, 0)]
    builder.add_node(*nodes[0])
    builder.add_node(*nodes[1])
    builder.add_node(*nodes[0])
    self.assertRaises(_mod_index.BadIndexDuplicateKey, builder.finish)