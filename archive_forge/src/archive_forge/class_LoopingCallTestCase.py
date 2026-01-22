import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
class LoopingCallTestCase(test_base.BaseTestCase):

    def setUp(self):
        super(LoopingCallTestCase, self).setUp()
        self.num_runs = 0

    def test_return_true(self):

        def _raise_it():
            raise loopingcall.LoopingCallDone(True)
        timer = loopingcall.FixedIntervalLoopingCall(_raise_it)
        self.assertTrue(timer.start(interval=0.5).wait())

    def test_monotonic_timer(self):

        def _raise_it():
            clock = eventlet.hubs.get_hub().clock
            ok = clock == time.monotonic
            raise loopingcall.LoopingCallDone(ok)
        timer = loopingcall.FixedIntervalLoopingCall(_raise_it)
        self.assertTrue(timer.start(interval=0.5).wait())

    def test_eventlet_clock(self):
        hub = eventlet.hubs.get_hub()
        self.assertEqual(time.monotonic, hub.clock)

    def test_return_false(self):

        def _raise_it():
            raise loopingcall.LoopingCallDone(False)
        timer = loopingcall.FixedIntervalLoopingCall(_raise_it)
        self.assertFalse(timer.start(interval=0.5).wait())

    def test_terminate_on_exception(self):

        def _raise_it():
            raise RuntimeError()
        timer = loopingcall.FixedIntervalLoopingCall(_raise_it)
        self.assertRaises(RuntimeError, timer.start(interval=0.5).wait)

    def _raise_and_then_done(self):
        if self.num_runs == 0:
            raise loopingcall.LoopingCallDone(False)
        else:
            self.num_runs = self.num_runs - 1
            raise RuntimeError()

    def test_do_not_stop_on_exception(self):
        self.useFixture(fixture.SleepFixture())
        self.num_runs = 2
        timer = loopingcall.FixedIntervalLoopingCall(self._raise_and_then_done)
        res = timer.start(interval=0.5, stop_on_exception=False).wait()
        self.assertFalse(res)

    def _wait_for_zero(self):
        """Called at an interval until num_runs == 0."""
        if self.num_runs == 0:
            raise loopingcall.LoopingCallDone(False)
        else:
            self.num_runs = self.num_runs - 1

    def test_no_double_start(self):
        wait_ev = greenthreading.Event()

        def _run_forever_until_set():
            if wait_ev.is_set():
                raise loopingcall.LoopingCallDone(True)
        timer = loopingcall.FixedIntervalLoopingCall(_run_forever_until_set)
        timer.start(interval=0.01)
        self.assertRaises(RuntimeError, timer.start, interval=0.01)
        wait_ev.set()
        timer.wait()

    def test_no_double_stop(self):

        def _raise_it():
            raise loopingcall.LoopingCallDone(False)
        timer = loopingcall.FixedIntervalLoopingCall(_raise_it)
        timer.start(interval=0.5)
        timer.stop()
        timer.stop()

    def test_repeat(self):
        self.useFixture(fixture.SleepFixture())
        self.num_runs = 2
        timer = loopingcall.FixedIntervalLoopingCall(self._wait_for_zero)
        self.assertFalse(timer.start(interval=0.5).wait())

    def assertAlmostEqual(self, expected, actual, precision=7, message=None):
        self.assertEqual(0, round(actual - expected, precision), message)

    @mock.patch('oslo_service.loopingcall.LoopingCallBase._sleep')
    @mock.patch('oslo_service.loopingcall.LoopingCallBase._elapsed')
    def test_interval_adjustment(self, elapsed_mock, sleep_mock):
        """Ensure the interval is adjusted to account for task duration."""
        self.num_runs = 3
        second = 1
        smidgen = 0.01
        elapsed_mock.side_effect = [second - smidgen, second + second, second + smidgen]
        timer = loopingcall.FixedIntervalLoopingCall(self._wait_for_zero)
        timer.start(interval=1.01).wait()
        expected_calls = [0.02, 0.0, 0.0]
        for i, call in enumerate(sleep_mock.call_args_list):
            expected = expected_calls[i]
            args, kwargs = call
            actual = args[0]
            message = 'Call #%d, expected: %s, actual: %s' % (i, expected, actual)
            self.assertAlmostEqual(expected, actual, message=message)

    def test_looping_call_timed_out(self):

        def _fake_task():
            pass
        timer = loopingcall.FixedIntervalWithTimeoutLoopingCall(_fake_task)
        self.assertRaises(loopingcall.LoopingCallTimeOut, timer.start(interval=0.1, timeout=0.3).wait)