import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
class DynamicLoopingCallTestCase(test_base.BaseTestCase):

    def setUp(self):
        super(DynamicLoopingCallTestCase, self).setUp()
        self.num_runs = 0

    def test_return_true(self):

        def _raise_it():
            raise loopingcall.LoopingCallDone(True)
        timer = loopingcall.DynamicLoopingCall(_raise_it)
        self.assertTrue(timer.start().wait())

    def test_monotonic_timer(self):

        def _raise_it():
            clock = eventlet.hubs.get_hub().clock
            ok = clock == time.monotonic
            raise loopingcall.LoopingCallDone(ok)
        timer = loopingcall.DynamicLoopingCall(_raise_it)
        self.assertTrue(timer.start().wait())

    def test_no_double_start(self):
        wait_ev = greenthreading.Event()

        def _run_forever_until_set():
            if wait_ev.is_set():
                raise loopingcall.LoopingCallDone(True)
            else:
                return 0.01
        timer = loopingcall.DynamicLoopingCall(_run_forever_until_set)
        timer.start()
        self.assertRaises(RuntimeError, timer.start)
        wait_ev.set()
        timer.wait()

    def test_return_false(self):

        def _raise_it():
            raise loopingcall.LoopingCallDone(False)
        timer = loopingcall.DynamicLoopingCall(_raise_it)
        self.assertFalse(timer.start().wait())

    def test_terminate_on_exception(self):

        def _raise_it():
            raise RuntimeError()
        timer = loopingcall.DynamicLoopingCall(_raise_it)
        self.assertRaises(RuntimeError, timer.start().wait)

    def _raise_and_then_done(self):
        if self.num_runs == 0:
            raise loopingcall.LoopingCallDone(False)
        else:
            self.num_runs = self.num_runs - 1
            raise RuntimeError()

    def test_do_not_stop_on_exception(self):
        self.useFixture(fixture.SleepFixture())
        self.num_runs = 2
        timer = loopingcall.DynamicLoopingCall(self._raise_and_then_done)
        timer.start(stop_on_exception=False).wait()

    def _wait_for_zero(self):
        """Called at an interval until num_runs == 0."""
        if self.num_runs == 0:
            raise loopingcall.LoopingCallDone(False)
        else:
            self.num_runs = self.num_runs - 1
            sleep_for = self.num_runs * 10 + 1
            return sleep_for

    def test_repeat(self):
        self.useFixture(fixture.SleepFixture())
        self.num_runs = 2
        timer = loopingcall.DynamicLoopingCall(self._wait_for_zero)
        self.assertFalse(timer.start().wait())

    def _timeout_task_without_any_return(self):
        pass

    def test_timeout_task_without_return_and_max_periodic(self):
        timer = loopingcall.DynamicLoopingCall(self._timeout_task_without_any_return)
        self.assertRaises(RuntimeError, timer.start().wait)

    def _timeout_task_without_return_but_with_done(self):
        if self.num_runs == 0:
            raise loopingcall.LoopingCallDone(False)
        else:
            self.num_runs = self.num_runs - 1

    @mock.patch('oslo_service.loopingcall.LoopingCallBase._sleep')
    def test_timeout_task_without_return(self, sleep_mock):
        self.num_runs = 1
        timer = loopingcall.DynamicLoopingCall(self._timeout_task_without_return_but_with_done)
        timer.start(periodic_interval_max=5).wait()
        sleep_mock.assert_has_calls([mock.call(5)])

    @mock.patch('oslo_service.loopingcall.LoopingCallBase._sleep')
    def test_interval_adjustment(self, sleep_mock):
        self.num_runs = 2
        timer = loopingcall.DynamicLoopingCall(self._wait_for_zero)
        timer.start(periodic_interval_max=5).wait()
        sleep_mock.assert_has_calls([mock.call(5), mock.call(1)])

    @mock.patch('oslo_service.loopingcall.LoopingCallBase._sleep')
    def test_initial_delay(self, sleep_mock):
        self.num_runs = 1
        timer = loopingcall.DynamicLoopingCall(self._wait_for_zero)
        timer.start(initial_delay=3).wait()
        sleep_mock.assert_has_calls([mock.call(3), mock.call(1)])