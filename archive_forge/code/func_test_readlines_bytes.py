from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_readlines_bytes(self):
    m = mock_open(read_data=b'abc\ndef\nghi\n')
    with patch('%s.open' % __name__, m, create=True):
        with open('abc', 'rb') as f:
            result = f.readlines()
    self.assertEqual(result, [b'abc\n', b'def\n', b'ghi\n'])