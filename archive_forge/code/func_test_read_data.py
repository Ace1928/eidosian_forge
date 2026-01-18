from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_read_data(self):
    mock = mock_open(read_data='foo')
    with patch('%s.open' % __name__, mock, create=True):
        h = open('bar')
        result = h.read()
    self.assertEqual(result, 'foo')