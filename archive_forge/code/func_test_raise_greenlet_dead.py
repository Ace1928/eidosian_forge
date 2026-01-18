from __future__ import print_function
from __future__ import absolute_import
import sys
import greenlet
from . import _test_extension
from . import TestCase
def test_raise_greenlet_dead(self):
    self.assertRaises(greenlet.GreenletExit, _test_extension.test_raise_dead_greenlet)