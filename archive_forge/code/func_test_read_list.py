from tempfile import NamedTemporaryFile
from testtools import TestCase
from subunit import read_test_list
from subunit.filters import find_stream
def test_read_list(self):
    with NamedTemporaryFile() as f:
        f.write(b'foo\nbar\n# comment\nother # comment\n')
        f.flush()
        self.assertEqual(read_test_list(f.name), ['foo', 'bar', 'other'])