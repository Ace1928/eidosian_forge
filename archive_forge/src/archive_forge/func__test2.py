import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def _test2(mock):
    mock.f(1)
    mock.f.assert_called_with(1)
    self.assertRaises(TypeError, mock.f)
    mock.g()
    mock.g.assert_called_with()
    self.assertRaises(TypeError, mock.g, 1)
    self.assertRaises(AttributeError, getattr, mock, 'h')
    mock.foo.lower()
    mock.foo.lower.assert_called_with()
    self.assertRaises(AttributeError, getattr, mock.foo, 'bar')
    mock.Bar()
    mock.Bar.assert_called_with()
    mock.Bar.a()
    mock.Bar.a.assert_called_with()
    self.assertRaises(TypeError, mock.Bar.a, 1)
    mock.Bar().a()
    mock.Bar().a.assert_called_with()
    self.assertRaises(TypeError, mock.Bar().a, 1)
    self.assertRaises(AttributeError, getattr, mock.Bar, 'b')
    self.assertRaises(AttributeError, getattr, mock.Bar(), 'b')