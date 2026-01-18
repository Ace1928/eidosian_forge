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
def test_switch_to_dead_greenlet_with_unstarted_perverse_parent(self):

    class Parent(RawGreenlet):

        def __getattribute__(self, name):
            if name == 'run':
                raise SomeError
    parent_never_started = Parent()
    seen = []
    child = RawGreenlet(lambda: seen.append(42), parent_never_started)
    with self.assertRaises(SomeError):
        child.switch()
    self.assertEqual(seen, [42])
    with self.assertRaises(SomeError):
        child.switch()
    self.assertEqual(seen, [42])