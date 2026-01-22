from tempfile import NamedTemporaryFile
from testtools import TestCase
from subunit import read_test_list
from subunit.filters import find_stream
class TestFindStream(TestCase):

    def test_no_argv(self):
        self.assertEqual('foo', find_stream('foo', []))

    def test_opens_file(self):
        f = NamedTemporaryFile()
        f.write(b'foo')
        f.flush()
        stream = find_stream('bar', [f.name])
        self.assertEqual(b'foo', stream.read())