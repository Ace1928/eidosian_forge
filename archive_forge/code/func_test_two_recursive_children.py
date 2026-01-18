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
def test_two_recursive_children(self):
    lst = []

    def f():
        lst.append('b')
        greenlet.getcurrent().parent.switch()

    def g():
        lst.append('a')
        g = RawGreenlet(f)
        g.switch()
        lst.append('c')
    g = RawGreenlet(g)
    self.assertEqual(sys.getrefcount(g), 2)
    g.switch()
    self.assertEqual(lst, ['a', 'b', 'c'])
    self.assertEqual(sys.getrefcount(g), 2)