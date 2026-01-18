from ... import errors, tests, transport
from .. import index as _mod_index
def test_parsing_non_adjacent_data_trims(self):
    index = self.make_index(nodes=self.make_nodes(64))
    result = index._lookup_keys_via_location([(index._size // 2, (b'40',))])
    self.assertEqual([((index._size // 2, (b'40',)), False)], result)
    self.assertEqual([(0, 4008), (5046, 8996)], index._parsed_byte_map)
    self.assertEqual([((), self.make_key(26)), (self.make_key(31), self.make_key(48))], index._parsed_key_map)