from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
class TestCaseWithStore(tests.TestCaseWithMemoryTransport):

    def get_chk_bytes(self):
        factory = groupcompress.make_pack_factory(False, False, 1)
        self.chk_bytes = factory(self.get_transport())
        return self.chk_bytes

    def _get_map(self, a_dict, maximum_size=0, chk_bytes=None, key_width=1, search_key_func=None):
        if chk_bytes is None:
            chk_bytes = self.get_chk_bytes()
        root_key = CHKMap.from_dict(chk_bytes, a_dict, maximum_size=maximum_size, key_width=key_width, search_key_func=search_key_func)
        root_key2 = CHKMap._create_via_map(chk_bytes, a_dict, maximum_size=maximum_size, key_width=key_width, search_key_func=search_key_func)
        self.assertEqual(root_key, root_key2, 'CHKMap.from_dict() did not match CHKMap._create_via_map')
        chkmap = CHKMap(chk_bytes, root_key, search_key_func=search_key_func)
        return chkmap

    def read_bytes(self, chk_bytes, key):
        stream = chk_bytes.get_record_stream([key], 'unordered', True)
        record = next(stream)
        if record.storage_kind == 'absent':
            self.fail('Store does not contain the key {}'.format(key))
        return record.get_bytes_as('fulltext')

    def to_dict(self, node, *args):
        return dict(node.iteritems(*args))