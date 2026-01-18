import threading
import time
from eventlet.green import threading as green_threading
import testscenarios
from testtools import testcase
import futurist
from futurist import rejection
from futurist.tests import base
def test_run_one(self):
    fut = self.executor.submit(returns_one)
    self.assertEqual(1, fut.result())
    self.assertTrue(fut.done())