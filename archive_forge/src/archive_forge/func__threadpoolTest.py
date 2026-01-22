import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def _threadpoolTest(self, method):
    """
        Test synchronization of calls made with C{method}, which should be
        one of the mechanisms of the threadpool to execute work in threads.
        """
    N = 10
    tp = threadpool.ThreadPool()
    tp.start()
    self.addCleanup(tp.stop)
    waiting = threading.Lock()
    waiting.acquire()
    actor = Synchronization(N, waiting)
    for i in range(N):
        method(tp, actor)
    self._waitForLock(waiting)
    self.assertFalse(actor.failures, f'run() re-entered {actor.failures} times')