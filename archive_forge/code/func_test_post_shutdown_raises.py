import threading
import time
from eventlet.green import threading as green_threading
import testscenarios
from testtools import testcase
import futurist
from futurist import rejection
from futurist.tests import base
def test_post_shutdown_raises(self):
    executor = self.executor_cls(**self.executor_kwargs)
    executor.shutdown()
    self.assertRaises(RuntimeError, executor.submit, returns_one)