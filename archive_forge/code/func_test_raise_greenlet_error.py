from __future__ import print_function
from __future__ import absolute_import
import sys
import greenlet
from . import _test_extension
from . import TestCase
def test_raise_greenlet_error(self):
    self.assertRaises(greenlet.error, _test_extension.test_raise_greenlet_error)