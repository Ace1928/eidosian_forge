import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def _waitForLock(self, lock):
    items = range(1000000)
    for i in items:
        if lock.acquire(False):
            break
        time.sleep(1e-05)
    else:
        self.fail('A long time passed without succeeding')