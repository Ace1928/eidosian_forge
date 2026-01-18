import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
def test_spec_class(self):

    class X(object):
        pass
    mock = Mock(spec=X)
    self.assertIsInstance(mock, X)
    mock = Mock(spec=X())
    self.assertIsInstance(mock, X)
    self.assertIs(mock.__class__, X)
    self.assertEqual(Mock().__class__.__name__, 'Mock')
    mock = Mock(spec_set=X)
    self.assertIsInstance(mock, X)
    mock = Mock(spec_set=X())
    self.assertIsInstance(mock, X)