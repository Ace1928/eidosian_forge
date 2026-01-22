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
def _only_test_some_versions(self):
    assert sys.version_info[0] >= 3
    if sys.version_info[:2] < (3, 8):
        self.skipTest('Only observed on 3.11')
    if RUNNING_ON_MANYLINUX:
        self.skipTest('Slow and not worth repeating here')