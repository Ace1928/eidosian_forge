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
@fails_leakcheck
def test_recursive_startup(self):

    class convoluted(RawGreenlet):

        def __init__(self):
            RawGreenlet.__init__(self)
            self.count = 0

        def __getattribute__(self, name):
            if name == 'run' and self.count == 0:
                self.count = 1
                self.switch(43)
            return RawGreenlet.__getattribute__(self, name)

        def run(self, value):
            while True:
                self.parent.switch(value)
    g = convoluted()
    self.assertEqual(g.switch(42), 43)
    self.expect_greenlet_leak = True