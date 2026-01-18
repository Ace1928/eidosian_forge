from ... import errors, tests, transport
from .. import index as _mod_index
def test_parsing_data_handles_parsed_contained_regions(self):
    index = self.make_index(nodes=self.make_nodes(128))
    result = index._lookup_keys_via_location([(index._size // 2, (b'40',))])
    self.assertEqual([(0, 4045), (11759, 15707)], index._parsed_byte_map)
    self.assertEqual([((), self.make_key(116)), (self.make_key(35), self.make_key(51))], index._parsed_key_map)
    result = index._lookup_keys_via_location([(11450, self.make_key(34)), (15707, self.make_key(52))])
    self.assertEqual([((11450, self.make_key(34)), (index, self.make_key(34), self.make_value(34))), ((15707, self.make_key(52)), (index, self.make_key(52), self.make_value(52)))], result)
    self.assertEqual([(0, 4045), (9889, 17993)], index._parsed_byte_map)