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
@ignores_leakcheck
def test_untracked_memory_doesnt_increase_unfinished_thread_dealloc_in_thread(self):
    self._check_untracked_memory_thread(deallocate_in_thread=True)