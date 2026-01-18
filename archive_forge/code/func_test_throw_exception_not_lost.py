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
def test_throw_exception_not_lost(self):

    class mygreenlet(RawGreenlet):

        def __getattribute__(self, name):
            try:
                raise Exception
            except:
                pass
            return RawGreenlet.__getattribute__(self, name)
    g = mygreenlet(lambda: None)
    self.assertRaises(SomeError, g.throw, SomeError())