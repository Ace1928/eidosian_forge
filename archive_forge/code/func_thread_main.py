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
def thread_main(greenlet_running_event):
    mine = MyGreenlet(greenlet_main)
    glets.append(mine)
    mine.switch()
    del mine
    greenlet_running_event.set()
    ref_cleared.wait(10)
    getcurrent()