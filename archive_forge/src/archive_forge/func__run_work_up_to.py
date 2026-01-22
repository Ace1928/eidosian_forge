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
def _run_work_up_to(self, stop_after_num_calls):
    ev = self.event_cls()
    called_tracker = []

    @periodics.periodic(0.1, run_immediately=True)
    def fast_periodic():
        called_tracker.append(True)
        if len(called_tracker) >= stop_after_num_calls:
            ev.set()
    callables = [(fast_periodic, None, None)]
    worker_kwargs = self.worker_kwargs.copy()
    w = periodics.PeriodicWorker(callables, **worker_kwargs)
    with self.create_destroy(w.start):
        ev.wait()
        w.stop()
    return (list(w.iter_watchers())[0], fast_periodic)