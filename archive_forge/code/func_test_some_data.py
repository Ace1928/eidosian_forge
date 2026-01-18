import zlib
from .. import chunk_writer
from . import TestCaseWithTransport
def test_some_data(self):
    writer = chunk_writer.ChunkWriter(4096)
    writer.write(b'foo bar baz quux\n')
    bytes_list, unused, padding = writer.finish()
    node_bytes = self.check_chunk(bytes_list, 4096)
    self.assertEqual(b'foo bar baz quux\n', node_bytes)
    self.assertEqual(None, unused)
    self.assertEqual(4073, padding)