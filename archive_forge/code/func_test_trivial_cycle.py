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
def test_trivial_cycle(self):
    glet = RawGreenlet(lambda: None)
    with self.assertRaises(ValueError) as exc:
        glet.parent = glet
    self.assertEqual(str(exc.exception), 'cyclic parent chain')