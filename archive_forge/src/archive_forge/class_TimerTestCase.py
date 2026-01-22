from oslo_messaging._drivers import common
from oslo_messaging import _utils as utils
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class TimerTestCase(test_utils.BaseTestCase):

    def test_no_duration_no_callback(self):
        t = common.DecayingTimer()
        t.start()
        remaining = t.check_return()
        self.assertIsNone(remaining)

    def test_no_duration_but_maximum(self):
        t = common.DecayingTimer()
        t.start()
        remaining = t.check_return(maximum=2)
        self.assertEqual(2, remaining)

    @mock.patch('oslo_utils.timeutils.now')
    def test_duration_expired_no_callback(self, now):
        now.return_value = 0
        t = common.DecayingTimer(2)
        t.start()
        now.return_value = 3
        remaining = t.check_return()
        self.assertEqual(0, remaining)

    @mock.patch('oslo_utils.timeutils.now')
    def test_duration_callback(self, now):
        now.return_value = 0
        t = common.DecayingTimer(2)
        t.start()
        now.return_value = 3
        callback = mock.Mock()
        remaining = t.check_return(callback)
        self.assertEqual(0, remaining)
        callback.assert_called_once_with()

    @mock.patch('oslo_utils.timeutils.now')
    def test_duration_callback_with_args(self, now):
        now.return_value = 0
        t = common.DecayingTimer(2)
        t.start()
        now.return_value = 3
        callback = mock.Mock()
        remaining = t.check_return(callback, 1, a='b')
        self.assertEqual(0, remaining)
        callback.assert_called_once_with(1, a='b')

    @mock.patch('oslo_utils.timeutils.now')
    def test_reset(self, now):
        now.return_value = 0
        t = common.DecayingTimer(3)
        t.start()
        now.return_value = 1
        remaining = t.check_return()
        self.assertEqual(2, remaining)
        t.restart()
        remaining = t.check_return()
        self.assertEqual(3, remaining)