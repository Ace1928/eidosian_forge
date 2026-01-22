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
class TestMainGreenlet(TestCase):

    def _check_current_is_main(self):
        assert 'main' in repr(greenlet.getcurrent())
        t = type(greenlet.getcurrent())
        assert 'main' not in repr(t)
        return t

    def test_main_greenlet_type_can_be_subclassed(self):
        main_type = self._check_current_is_main()
        subclass = type('subclass', (main_type,), {})
        self.assertIsNotNone(subclass)

    def test_main_greenlet_is_greenlet(self):
        self._check_current_is_main()
        self.assertIsInstance(greenlet.getcurrent(), RawGreenlet)