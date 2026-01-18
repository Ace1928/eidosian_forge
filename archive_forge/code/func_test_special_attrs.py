import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_special_attrs(self):

    def foo(x=0):
        """TEST"""
        return x
    with patch.object(foo, '__defaults__', (1,)):
        self.assertEqual(foo(), 1)
    self.assertEqual(foo(), 0)
    with patch.object(foo, '__doc__', 'FUN'):
        self.assertEqual(foo.__doc__, 'FUN')
    self.assertEqual(foo.__doc__, 'TEST')
    with patch.object(foo, '__module__', 'testpatch2'):
        self.assertEqual(foo.__module__, 'testpatch2')
    self.assertEqual(foo.__module__, __name__)
    if hasattr(self.test_special_attrs, '__annotations__'):
        with patch.object(foo, '__annotations__', dict([('s', 1)])):
            self.assertEqual(foo.__annotations__, dict([('s', 1)]))
        self.assertEqual(foo.__annotations__, dict())
    if hasattr(self.test_special_attrs, '__kwdefaults__'):
        foo = eval('lambda *a, x=0: x')
        with patch.object(foo, '__kwdefaults__', dict([('x', 1)])):
            self.assertEqual(foo(), 1)
        self.assertEqual(foo(), 0)