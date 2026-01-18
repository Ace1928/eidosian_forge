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
def test_parent_restored_on_kill(self):
    hub = RawGreenlet(lambda: None)
    main = greenlet.getcurrent()
    result = []

    def worker():
        try:
            main.switch()
        except greenlet.GreenletExit:
            result.append(greenlet.getcurrent().parent)
            result.append(greenlet.getcurrent())
            hub.switch()
    g = RawGreenlet(worker, parent=hub)
    g.switch()
    del g
    self.assertTrue(result)
    self.assertIs(result[0], main)
    self.assertIs(result[1].parent, hub)
    del result[:]
    hub = None
    main = None