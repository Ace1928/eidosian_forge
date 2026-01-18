from ... import errors, tests, transport
from .. import index as _mod_index
def test_validate_missing_end_line_nonempty(self):
    index = self.make_index(2, nodes=[((b'key',), b'', ([], []))])
    trans = self.get_transport()
    content = trans.get_bytes('index')
    trans.put_bytes('index', content[:-1])
    self.assertRaises(_mod_index.BadIndexData, index.validate)