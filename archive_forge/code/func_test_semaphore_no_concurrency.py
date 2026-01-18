import queue
from threading import Thread
from timeit import default_timer as timer
from unittest import mock
import testtools
from keystoneauth1 import _fair_semaphore
def test_semaphore_no_concurrency(self):
    start = timer()
    self._concurrency_core(None, 0.1)
    end = timer()
    self.assertTrue(end - start > 1.0)
    self.assertEqual(self.mock_payload.do_something.call_count, 10)