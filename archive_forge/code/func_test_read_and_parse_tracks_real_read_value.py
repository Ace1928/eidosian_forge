from ... import errors, tests, transport
from .. import index as _mod_index
def test_read_and_parse_tracks_real_read_value(self):
    index = self.make_index(nodes=self.make_nodes(10))
    del index._transport._activity[:]
    index._read_and_parse([(0, 200)])
    self.assertEqual([('readv', 'index', [(0, 200)], True, index._size)], index._transport._activity)
    self.assertEqual(index._size, index._bytes_read)