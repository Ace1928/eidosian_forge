from ... import errors, tests, transport
from .. import index as _mod_index
def test_open_bad_index_no_error(self):
    trans = self.get_transport()
    trans.put_bytes('name', b'not an index\n')
    idx = _mod_index.GraphIndex(trans, 'name', 13)