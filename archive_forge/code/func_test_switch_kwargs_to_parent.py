from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gc
import sys
import time
import threading
from abc import ABCMeta, abstractmethod
import greenlet
from greenlet import greenlet as RawGreenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def test_switch_kwargs_to_parent(self):

    def run(x):
        greenlet.getcurrent().parent.switch(x=x)
        greenlet.getcurrent().parent.switch(2, x=3)
        return (x, x ** 2)
    g = RawGreenlet(run)
    self.assertEqual({'x': 3}, g.switch(3))
    self.assertEqual(((2,), {'x': 3}), g.switch())
    self.assertEqual((3, 9), g.switch())