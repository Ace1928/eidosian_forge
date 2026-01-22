import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def _check_someclass_mock(self, mock):
    self.assertRaises(AttributeError, getattr, mock, 'foo')
    mock.one(1, 2)
    mock.one.assert_called_with(1, 2)
    self.assertRaises(AssertionError, mock.one.assert_called_with, 3, 4)
    self.assertRaises(TypeError, mock.one, 1)
    mock.two()
    mock.two.assert_called_with()
    self.assertRaises(AssertionError, mock.two.assert_called_with, 3)
    self.assertRaises(TypeError, mock.two, 1)
    mock.three()
    mock.three.assert_called_with()
    self.assertRaises(AssertionError, mock.three.assert_called_with, 3)
    self.assertRaises(TypeError, mock.three, 3, 2)
    mock.three(1)
    mock.three.assert_called_with(1)
    mock.three(a=1)
    mock.three.assert_called_with(a=1)