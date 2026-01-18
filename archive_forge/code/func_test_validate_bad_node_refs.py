from ... import errors, tests, transport
from .. import index as _mod_index
def test_validate_bad_node_refs(self):
    idx = self.make_index(2)
    trans = self.get_transport()
    content = trans.get_bytes('index')
    new_content = content[:-2] + b'a\n\n'
    trans.put_bytes('index', new_content)
    self.assertRaises(_mod_index.BadIndexOptions, idx.validate)