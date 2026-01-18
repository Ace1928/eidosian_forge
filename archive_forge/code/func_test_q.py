import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def test_q(self) -> None:
    """
        There is a property '_queue' for legacy purposes
        """
    pool = threadpool.ThreadPool(0, 1)
    self.assertEqual(pool._queue.qsize(), 0)