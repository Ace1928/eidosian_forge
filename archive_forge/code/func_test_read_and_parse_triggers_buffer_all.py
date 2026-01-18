from ... import errors, tests, transport
from .. import index as _mod_index
def test_read_and_parse_triggers_buffer_all(self):
    index = self.make_index(key_elements=2, nodes=[((b'name', b'fin1'), b'data', ()), ((b'name', b'fin2'), b'beta', ()), ((b'ref', b'erence'), b'refdata', ())])
    self.assertTrue(index._size > 0)
    self.assertIs(None, index._nodes)
    index._read_and_parse([(0, index._size)])
    self.assertIsNot(None, index._nodes)