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
def test_waiting_immediate_add_processed(self):
    ran_at = []

    @periodics.periodic(0.1, run_immediately=True)
    def activated_periodic():
        ran_at.append(time.time())
    w = periodics.PeriodicWorker([], **self.worker_kwargs)
    with self.create_destroy(w.start, allow_empty=True):
        self.sleep(0.5)
        w.add(activated_periodic)
        while len(ran_at) == 0:
            self.sleep(0.1)
        w.stop()