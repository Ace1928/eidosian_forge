import threading
import time
from eventlet.green import threading as green_threading
import testscenarios
from testtools import testcase
import futurist
from futurist import rejection
from futurist.tests import base
def test_restartable(self):
    if not self.restartable:
        raise testcase.TestSkipped('not restartable')
    else:
        executor = self.executor_cls(**self.executor_kwargs)
        fut = executor.submit(returns_one)
        self.assertEqual(1, fut.result())
        executor.shutdown()
        self.assertEqual(1, executor.statistics.executed)
        self.assertRaises(RuntimeError, executor.submit, returns_one)
        executor.restart()
        self.assertEqual(0, executor.statistics.executed)
        fut = executor.submit(returns_one)
        self.assertEqual(1, fut.result())
        self.assertEqual(1, executor.statistics.executed)
        executor.shutdown()