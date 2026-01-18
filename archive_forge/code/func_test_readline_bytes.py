from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_readline_bytes(self):
    m = mock_open(read_data=b'abc\ndef\nghi\n')
    with patch('%s.open' % __name__, m, create=True):
        with open('abc', 'rb') as f:
            line1 = f.readline()
            line2 = f.readline()
            line3 = f.readline()
    self.assertEqual(line1, b'abc\n')
    self.assertEqual(line2, b'def\n')
    self.assertEqual(line3, b'ghi\n')