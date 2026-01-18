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
def test_threaded_updatecurrent(self):
    lock1 = threading.Lock()
    lock1.acquire()
    lock2 = threading.Lock()
    lock2.acquire()

    class finalized(object):

        def __del__(self):
            lock2.release()
            lock1.acquire()

    def deallocator():
        greenlet.getcurrent().parent.switch()

    def fthread():
        lock2.acquire()
        greenlet.getcurrent()
        del g[0]
        lock1.release()
        lock2.acquire()
        greenlet.getcurrent()
        lock1.release()
    main = greenlet.getcurrent()
    g = [RawGreenlet(deallocator)]
    g[0].bomb = finalized()
    g[0].switch()
    t = threading.Thread(target=fthread)
    t.start()
    lock2.release()
    lock1.acquire()
    self.assertEqual(greenlet.getcurrent(), main)
    t.join(10)