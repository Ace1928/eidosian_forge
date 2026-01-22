import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def assertFlattened(self, expected, key, value, refs):
    flat_key, flat_line = self.parse_btree._flatten_node((None, key, value, refs), bool(refs))
    self.assertEqual(b'\x00'.join(key), flat_key)
    self.assertEqual(expected, flat_line)