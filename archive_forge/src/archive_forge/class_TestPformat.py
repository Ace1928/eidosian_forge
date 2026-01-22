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
class TestPformat(base.TestCase):

    def test_invalid(self):

        @periodics.periodic(1)
        def a():
            pass

        @periodics.periodic(2)
        def b():
            pass

        @periodics.periodic(3)
        def c():
            pass
        callables = [(a, None, None), (b, None, None), (c, None, None)]
        w = periodics.PeriodicWorker(callables)
        self.assertRaises(ValueError, w.pformat, columns=[])
        self.assertRaises(ValueError, w.pformat, columns=['not a column'])