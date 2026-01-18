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
def test_switch_to_another_thread(self):
    data = {}
    created_event = threading.Event()
    done_event = threading.Event()

    def run():
        data['g'] = RawGreenlet(lambda: None)
        created_event.set()
        done_event.wait(10)
    thread = threading.Thread(target=run)
    thread.start()
    created_event.wait(10)
    with self.assertRaises(greenlet.error):
        data['g'].switch()
    done_event.set()
    thread.join(10)
    data.clear()