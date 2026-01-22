import time
from unittest import mock
import eventlet
from eventlet.green import threading as greenthreading
from oslotest import base as test_base
from oslo_service import fixture
from oslo_service import loopingcall
class TestBackOffLoopingCall(test_base.BaseTestCase):

    @mock.patch('random.SystemRandom.gauss')
    @mock.patch('oslo_service.loopingcall.LoopingCallBase._sleep')
    def test_exponential_backoff(self, sleep_mock, random_mock):

        def false():
            return False
        random_mock.return_value = 0.8
        self.assertRaises(loopingcall.LoopingCallTimeOut, loopingcall.BackOffLoopingCall(false).start().wait)
        expected_times = [mock.call(1.6), mock.call(2.4000000000000004), mock.call(3.6), mock.call(5.4), mock.call(8.1), mock.call(12.15), mock.call(18.225), mock.call(27.337500000000002), mock.call(41.00625), mock.call(61.509375000000006), mock.call(92.26406250000001)]
        self.assertEqual(expected_times, sleep_mock.call_args_list)

    @mock.patch('random.SystemRandom.gauss')
    @mock.patch('oslo_service.loopingcall.LoopingCallBase._sleep')
    def test_exponential_backoff_negative_value(self, sleep_mock, random_mock):

        def false():
            return False
        random_mock.return_value = -0.8
        self.assertRaises(loopingcall.LoopingCallTimeOut, loopingcall.BackOffLoopingCall(false).start().wait)
        expected_times = [mock.call(1.6), mock.call(2.4000000000000004), mock.call(3.6), mock.call(5.4), mock.call(8.1), mock.call(12.15), mock.call(18.225), mock.call(27.337500000000002), mock.call(41.00625), mock.call(61.509375000000006), mock.call(92.26406250000001)]
        self.assertEqual(expected_times, sleep_mock.call_args_list)

    @mock.patch('random.SystemRandom.gauss')
    @mock.patch('oslo_service.loopingcall.LoopingCallBase._sleep')
    def test_no_backoff(self, sleep_mock, random_mock):
        random_mock.return_value = 1
        func = mock.Mock()
        func.side_effect = [True, True, True, loopingcall.LoopingCallDone(retvalue='return value')]
        retvalue = loopingcall.BackOffLoopingCall(func).start().wait()
        expected_times = [mock.call(1), mock.call(1), mock.call(1)]
        self.assertEqual(expected_times, sleep_mock.call_args_list)
        self.assertEqual('return value', retvalue)

    @mock.patch('random.SystemRandom.gauss')
    @mock.patch('oslo_service.loopingcall.LoopingCallBase._sleep')
    def test_no_sleep(self, sleep_mock, random_mock):
        random_mock.return_value = 1
        func = mock.Mock()
        func.side_effect = loopingcall.LoopingCallDone(retvalue='return value')
        retvalue = loopingcall.BackOffLoopingCall(func).start().wait()
        self.assertFalse(sleep_mock.called)
        self.assertEqual('return value', retvalue)

    @mock.patch('random.SystemRandom.gauss')
    @mock.patch('oslo_service.loopingcall.LoopingCallBase._sleep')
    def test_max_interval(self, sleep_mock, random_mock):

        def false():
            return False
        random_mock.return_value = 0.8
        self.assertRaises(loopingcall.LoopingCallTimeOut, loopingcall.BackOffLoopingCall(false).start(max_interval=60).wait)
        expected_times = [mock.call(1.6), mock.call(2.4000000000000004), mock.call(3.6), mock.call(5.4), mock.call(8.1), mock.call(12.15), mock.call(18.225), mock.call(27.337500000000002), mock.call(41.00625), mock.call(60), mock.call(60), mock.call(60)]
        self.assertEqual(expected_times, sleep_mock.call_args_list)