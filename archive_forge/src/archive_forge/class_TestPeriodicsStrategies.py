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
class TestPeriodicsStrategies(base.TestCase):

    def test_invalids(self):
        self.assertRaises(ValueError, periodics.PeriodicWorker, [], schedule_strategy='not_a_strategy')