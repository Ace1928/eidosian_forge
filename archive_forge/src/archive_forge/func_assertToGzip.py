import gzip
from io import BytesIO
from .. import tests, tuned_gzip
def assertToGzip(self, chunks):
    raw_bytes = b''.join(chunks)
    gzfromchunks = tuned_gzip.chunks_to_gzip(chunks)
    decoded = gzip.GzipFile(fileobj=BytesIO(b''.join(gzfromchunks))).read()
    lraw, ldecoded = (len(raw_bytes), len(decoded))
    self.assertEqual(lraw, ldecoded, 'Expecting data length %d, got %d' % (lraw, ldecoded))
    self.assertEqual(raw_bytes, decoded)