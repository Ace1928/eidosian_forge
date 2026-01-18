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
def test_send_exception(self):
    seen = []
    g1 = RawGreenlet(fmain)
    g1.switch(seen)
    self.assertRaises(KeyError, send_exception, g1, KeyError)
    self.assertEqual(seen, [KeyError])