from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_readline_data(self):
    mock = mock_open(read_data='foo\nbar\nbaz\n')
    with patch('%s.open' % __name__, mock, create=True):
        h = open('bar')
        line1 = h.readline()
        line2 = h.readline()
        line3 = h.readline()
    self.assertEqual(line1, 'foo\n')
    self.assertEqual(line2, 'bar\n')
    self.assertEqual(line3, 'baz\n')
    mock = mock_open(read_data='foo')
    with patch('%s.open' % __name__, mock, create=True):
        h = open('bar')
        result = h.readline()
    self.assertEqual(result, 'foo')