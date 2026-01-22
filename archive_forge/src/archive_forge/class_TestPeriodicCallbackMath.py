import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
from collections.abc import Generator
import contextlib
import datetime
import functools
import socket
import subprocess
import sys
import threading
import time
import types
from unittest import mock
import unittest
from tornado.escape import native_str
from tornado import gen
from tornado.ioloop import IOLoop, TimeoutError, PeriodicCallback
from tornado.log import app_log
from tornado.testing import (
from tornado.test.util import (
from tornado.concurrent import Future
import typing
class TestPeriodicCallbackMath(unittest.TestCase):

    def simulate_calls(self, pc, durations):
        """Simulate a series of calls to the PeriodicCallback.

        Pass a list of call durations in seconds (negative values
        work to simulate clock adjustments during the call, or more or
        less equivalently, between calls). This method returns the
        times at which each call would be made.
        """
        calls = []
        now = 1000
        pc._next_timeout = now
        for d in durations:
            pc._update_next(now)
            calls.append(pc._next_timeout)
            now = pc._next_timeout + d
        return calls

    def dummy(self):
        pass

    def test_basic(self):
        pc = PeriodicCallback(self.dummy, 10000)
        self.assertEqual(self.simulate_calls(pc, [0] * 5), [1010, 1020, 1030, 1040, 1050])

    def test_overrun(self):
        call_durations = [9, 9, 10, 11, 20, 20, 35, 35, 0, 0, 0]
        expected = [1010, 1020, 1030, 1050, 1070, 1100, 1130, 1170, 1210, 1220, 1230]
        pc = PeriodicCallback(self.dummy, 10000)
        self.assertEqual(self.simulate_calls(pc, call_durations), expected)

    def test_clock_backwards(self):
        pc = PeriodicCallback(self.dummy, 10000)
        self.assertEqual(self.simulate_calls(pc, [-2, -1, -3, -2, 0]), [1010, 1020, 1030, 1040, 1050])
        self.assertEqual(self.simulate_calls(pc, [-100, 0, 0]), [1010, 1020, 1030])

    def test_jitter(self):
        random_times = [0.5, 1, 0, 0.75]
        expected = [1010, 1022.5, 1030, 1041.25]
        call_durations = [0] * len(random_times)
        pc = PeriodicCallback(self.dummy, 10000, jitter=0.5)

        def mock_random():
            return random_times.pop(0)
        with mock.patch('random.random', mock_random):
            self.assertEqual(self.simulate_calls(pc, call_durations), expected)

    def test_timedelta(self):
        pc = PeriodicCallback(lambda: None, datetime.timedelta(minutes=1, seconds=23))
        expected_callback_time = 83000
        self.assertEqual(pc.callback_time, expected_callback_time)