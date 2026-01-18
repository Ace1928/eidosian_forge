from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_with_statement_exception(self):
    try:
        with patch('%s.something' % __name__, sentinel.Something2):
            self.assertEqual(something, sentinel.Something2, 'unpatched')
            raise Exception('pow')
    except Exception:
        pass
    else:
        self.fail('patch swallowed exception')
    self.assertEqual(something, sentinel.Something)