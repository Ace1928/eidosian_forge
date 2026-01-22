from __future__ import print_function, absolute_import, division
import sys
import gc
import time
import weakref
import threading
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
from .leakcheck import ignores_leakcheck
from .leakcheck import RUNNING_ON_MANYLINUX
def assertClocksUsed(self):
    used = greenlet._greenlet.get_clocks_used_doing_optional_cleanup()
    self.assertGreaterEqual(used, 0)
    greenlet._greenlet.enable_optional_cleanup(True)
    used2 = greenlet._greenlet.get_clocks_used_doing_optional_cleanup()
    self.assertEqual(used, used2)
    self.assertGreater(greenlet._greenlet.CLOCKS_PER_SEC, 1)