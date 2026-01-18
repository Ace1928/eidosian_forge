import contextlib
import functools
import threading
import time
from unittest import mock
import eventlet
from eventlet.green import threading as green_threading
import testscenarios
import futurist
from futurist import periodics
from futurist.tests import base
def test_worker(self):
    called = []

    def cb():
        called.append(1)
    callables = [(every_one_sec, (cb,), None), (every_half_sec, (cb,), None)]
    executor = self.executor_cls(**self.executor_kwargs)
    executor_factory = lambda: executor
    w = periodics.PeriodicWorker(callables, executor_factory=executor_factory, **self.worker_kwargs)
    with self.create_destroy(w.start):
        self.sleep(2.0)
        w.stop()
    am_called = sum(called)
    self.assertGreaterEqual(am_called, 4)
    self.assertFalse(executor.alive)