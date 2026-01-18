from ... import errors, tests, transport
from .. import index as _mod_index
def test_open_sets_parsed_map_empty(self):
    index = self.make_index()
    self.assertEqual([], index._parsed_byte_map)
    self.assertEqual([], index._parsed_key_map)