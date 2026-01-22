import gzip
from io import BytesIO
from .. import tests, tuned_gzip
class TestToGzip(tests.TestCase):

    def assertToGzip(self, chunks):
        raw_bytes = b''.join(chunks)
        gzfromchunks = tuned_gzip.chunks_to_gzip(chunks)
        decoded = gzip.GzipFile(fileobj=BytesIO(b''.join(gzfromchunks))).read()
        lraw, ldecoded = (len(raw_bytes), len(decoded))
        self.assertEqual(lraw, ldecoded, 'Expecting data length %d, got %d' % (lraw, ldecoded))
        self.assertEqual(raw_bytes, decoded)

    def test_single_chunk(self):
        self.assertToGzip([b'a modest chunk\nwith some various\nbits\n'])

    def test_simple_text(self):
        self.assertToGzip([b'some\n', b'strings\n', b'to\n', b'process\n'])

    def test_large_chunks(self):
        self.assertToGzip([b'a large string\n' * 1024])
        self.assertToGzip([b'a large string\n'] * 1024)

    def test_enormous_chunks(self):
        self.assertToGzip([b'a large string\n' * 1024 * 256])
        self.assertToGzip([b'a large string\n'] * 1024 * 256)