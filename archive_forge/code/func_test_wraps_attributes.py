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
def test_wraps_attributes(self):

    class Real(object):
        attribute = Mock()
    real = Real()
    mock = Mock(wraps=real)
    self.assertEqual(mock.attribute(), real.attribute())
    self.assertRaises(AttributeError, lambda: mock.fish)
    self.assertNotEqual(mock.attribute, real.attribute)
    result = mock.attribute.frog(1, 2, fish=3)
    Real.attribute.frog.assert_called_with(1, 2, fish=3)
    self.assertEqual(result, Real.attribute.frog())