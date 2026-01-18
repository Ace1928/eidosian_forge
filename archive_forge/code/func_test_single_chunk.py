import gzip
from io import BytesIO
from .. import tests, tuned_gzip
def test_single_chunk(self):
    self.assertToGzip([b'a modest chunk\nwith some various\nbits\n'])