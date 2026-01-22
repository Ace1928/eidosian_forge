import queue
from threading import Thread
from timeit import default_timer as timer
from unittest import mock
import testtools
from keystoneauth1 import _fair_semaphore
def _concurrency_core(self, concurrency, delay):
    self.s = _fair_semaphore.FairSemaphore(concurrency, delay)
    self.q = queue.Queue()
    for i in range(5):
        t = Thread(target=self._thread_worker)
        t.daemon = True
        t.start()
    for item in range(0, 10):
        self.q.put(item)
    self.q.join()