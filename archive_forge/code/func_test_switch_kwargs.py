from __future__ import print_function
from __future__ import absolute_import
import sys
import greenlet
from . import _test_extension
from . import TestCase
def test_switch_kwargs(self):

    def adder(x, y):
        return x * y
    g = greenlet.greenlet(adder)
    self.assertEqual(6, _test_extension.test_switch_kwargs(g, x=3, y=2))