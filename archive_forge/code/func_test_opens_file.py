from tempfile import NamedTemporaryFile
from testtools import TestCase
from subunit import read_test_list
from subunit.filters import find_stream
def test_opens_file(self):
    f = NamedTemporaryFile()
    f.write(b'foo')
    f.flush()
    stream = find_stream('bar', [f.name])
    self.assertEqual(b'foo', stream.read())