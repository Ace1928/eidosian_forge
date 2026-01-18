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
def test_watcher(self):

    def cb():
        pass
    callables = [(every_one_sec, (cb,), None), (every_half_sec, (cb,), None)]
    executor_factory = lambda: self.executor_cls(**self.executor_kwargs)
    w = periodics.PeriodicWorker(callables, executor_factory=executor_factory, **self.worker_kwargs)
    with self.create_destroy(w.start):
        self.sleep(2.0)
        w.stop()
    for watcher in w.iter_watchers():
        self.assertGreaterEqual(watcher.runs, 1)
    w.reset()
    for watcher in w.iter_watchers():
        self.assertEqual(watcher.runs, 0)
        self.assertEqual(watcher.successes, 0)
        self.assertEqual(watcher.failures, 0)
        self.assertEqual(watcher.elapsed, 0)
        self.assertEqual(watcher.elapsed_waiting, 0)