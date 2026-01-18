from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_with_statement_imbricated(self):
    with patch('%s.something' % __name__) as mock_something:
        self.assertEqual(something, mock_something, 'unpatched')
        with patch('%s.something_else' % __name__) as mock_something_else:
            self.assertEqual(something_else, mock_something_else, 'unpatched')
    self.assertEqual(something, sentinel.Something)
    self.assertEqual(something_else, sentinel.SomethingElse)