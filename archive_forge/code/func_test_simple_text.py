import gzip
from io import BytesIO
from .. import tests, tuned_gzip
def test_simple_text(self):
    self.assertToGzip([b'some\n', b'strings\n', b'to\n', b'process\n'])