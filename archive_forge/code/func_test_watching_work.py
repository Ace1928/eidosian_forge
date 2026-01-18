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
def test_watching_work(self):
    for i in [3, 5, 9, 11]:
        watcher, cb = self._run_work_up_to(i)
        self.assertEqual(cb, watcher.work.callback)
        self.assertGreaterEqual(i, watcher.runs)
        self.assertGreaterEqual(i, watcher.successes)
        self.assertEqual((), watcher.work.args)
        self.assertEqual({}, watcher.work.kwargs)