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
class TestRetrySubmission(base.TestCase):

    def test_retry_submission(self):
        called = []

        def cb():
            called.append(1)
        callables = [(every_one_sec, (cb,), None), (every_half_sec, (cb,), None)]
        w = periodics.PeriodicWorker(callables, executor_factory=RejectingExecutor, cond_cls=green_threading.Condition, event_cls=green_threading.Event)
        w._RESCHEDULE_DELAY = 0
        w._RESCHEDULE_JITTER = 0
        with create_destroy_green_thread(w.start):
            eventlet.sleep(2.0)
            w.stop()
        am_called = sum(called)
        self.assertGreaterEqual(am_called, 4)