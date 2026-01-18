import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_skip_attributeerrors(self):

    class Raiser(object):

        def __get__(self, obj, type=None):
            if obj is None:
                raise AttributeError('Can only be accessed via an instance')

    class RaiserClass(object):
        raiser = Raiser()

        @staticmethod
        def existing(a, b):
            return a + b
    s = create_autospec(RaiserClass)
    self.assertRaises(TypeError, lambda x: s.existing(1, 2, 3))
    s.existing(1, 2)
    self.assertRaises(AttributeError, lambda: s.nonexisting)
    obj = s.raiser
    (obj.foo, obj.bar)