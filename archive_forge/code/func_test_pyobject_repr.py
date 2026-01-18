from ctypes import *
import unittest
from test import support
from _ctypes import PyObj_FromPtr
from sys import getrefcount as grc
def test_pyobject_repr(self):
    self.assertEqual(repr(py_object()), 'py_object(<NULL>)')
    self.assertEqual(repr(py_object(42)), 'py_object(42)')
    self.assertEqual(repr(py_object(object)), 'py_object(%r)' % object)