import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
class ForeverRetryUncaughtExceptionsTest(test_base.BaseTestCase):

    def setUp(self):
        super(ForeverRetryUncaughtExceptionsTest, self).setUp()
        self._exceptions = []
        self.useFixture(fixtures.MockPatch('time.sleep', return_value=None))

    @excutils.forever_retry_uncaught_exceptions
    def exception_generator(self):
        while self._exceptions:
            raise self._exceptions.pop(0)

    @mock.patch.object(logging, 'exception')
    @mock.patch.object(timeutils, 'now')
    def test_exc_retrier_1exc_gives_1log(self, mock_now, mock_log):
        self._exceptions = [Exception('unexpected %d' % 1)]
        mock_now.side_effect = [0]
        self.exception_generator()
        self.assertEqual([], self._exceptions)
        mock_log.assert_called_once_with('Unexpected exception occurred %d time(s)... retrying.' % 1)
        mock_now.assert_has_calls([mock.call()])

    @mock.patch.object(logging, 'exception')
    @mock.patch.object(timeutils, 'now')
    def test_exc_retrier_same_10exc_1min_gives_1log(self, mock_now, mock_log):
        self._exceptions = [Exception('unexpected 1')]
        mock_now_side_effect = [0]
        for i in range(2, 11):
            self._exceptions.append(Exception('unexpected 1'))
            mock_now_side_effect.append(i)
        mock_now.side_effect = mock_now_side_effect
        self.exception_generator()
        self.assertEqual([], self._exceptions)
        self.assertEqual(10, len(mock_now.mock_calls))
        self.assertEqual(1, len(mock_log.mock_calls))
        mock_log.assert_has_calls([mock.call('Unexpected exception occurred 1 time(s)... retrying.')])

    @mock.patch.object(logging, 'exception')
    @mock.patch.object(timeutils, 'now')
    def test_exc_retrier_same_2exc_2min_gives_2logs(self, mock_now, mock_log):
        self._exceptions = [Exception('unexpected 1'), Exception('unexpected 1')]
        mock_now.side_effect = [0, 65, 65, 66]
        self.exception_generator()
        self.assertEqual([], self._exceptions)
        self.assertEqual(4, len(mock_now.mock_calls))
        self.assertEqual(2, len(mock_log.mock_calls))
        mock_log.assert_has_calls([mock.call('Unexpected exception occurred 1 time(s)... retrying.'), mock.call('Unexpected exception occurred 1 time(s)... retrying.')])

    @mock.patch.object(logging, 'exception')
    @mock.patch.object(timeutils, 'now')
    def test_exc_retrier_same_10exc_2min_gives_2logs(self, mock_now, mock_log):
        self._exceptions = [Exception('unexpected 1')]
        mock_now_side_effect = [0]
        for ts in [12, 23, 34, 45]:
            self._exceptions.append(Exception('unexpected 1'))
            mock_now_side_effect.append(ts)
        self._exceptions.append(Exception('unexpected 1'))
        mock_now_side_effect.append(106)
        for ts in [106, 107]:
            mock_now_side_effect.append(ts)
        for ts in [117, 128, 139, 150]:
            self._exceptions.append(Exception('unexpected 1'))
            mock_now_side_effect.append(ts)
        mock_now.side_effect = mock_now_side_effect
        self.exception_generator()
        self.assertEqual([], self._exceptions)
        self.assertEqual(12, len(mock_now.mock_calls))
        self.assertEqual(2, len(mock_log.mock_calls))
        mock_log.assert_has_calls([mock.call('Unexpected exception occurred 1 time(s)... retrying.'), mock.call('Unexpected exception occurred 5 time(s)... retrying.')])

    @mock.patch.object(logging, 'exception')
    @mock.patch.object(timeutils, 'now')
    def test_exc_retrier_mixed_4exc_1min_gives_2logs(self, mock_now, mock_log):
        self._exceptions = [Exception('unexpected 1')]
        mock_now_side_effect = [0]
        self._exceptions.append(Exception('unexpected 1'))
        mock_now_side_effect.append(5)
        self._exceptions.append(Exception('unexpected 2'))
        mock_now_side_effect.extend([10, 20])
        self._exceptions.append(Exception('unexpected 2'))
        mock_now_side_effect.append(25)
        mock_now.side_effect = mock_now_side_effect
        self.exception_generator()
        self.assertEqual([], self._exceptions)
        self.assertEqual(5, len(mock_now.mock_calls))
        self.assertEqual(2, len(mock_log.mock_calls))
        mock_log.assert_has_calls([mock.call('Unexpected exception occurred 1 time(s)... retrying.'), mock.call('Unexpected exception occurred 1 time(s)... retrying.')])

    @mock.patch.object(logging, 'exception')
    @mock.patch.object(timeutils, 'now')
    def test_exc_retrier_mixed_4exc_2min_gives_2logs(self, mock_now, mock_log):
        self._exceptions = [Exception('unexpected 1')]
        mock_now_side_effect = [0]
        self._exceptions.append(Exception('unexpected 1'))
        mock_now_side_effect.append(10)
        self._exceptions.append(Exception('unexpected 2'))
        mock_now_side_effect.extend([100, 105])
        self._exceptions.append(Exception('unexpected 2'))
        mock_now_side_effect.append(110)
        mock_now.side_effect = mock_now_side_effect
        self.exception_generator()
        self.assertEqual([], self._exceptions)
        self.assertEqual(5, len(mock_now.mock_calls))
        self.assertEqual(2, len(mock_log.mock_calls))
        mock_log.assert_has_calls([mock.call('Unexpected exception occurred 1 time(s)... retrying.'), mock.call('Unexpected exception occurred 1 time(s)... retrying.')])

    @mock.patch.object(logging, 'exception')
    @mock.patch.object(timeutils, 'now')
    def test_exc_retrier_mixed_4exc_2min_gives_3logs(self, mock_now, mock_log):
        self._exceptions = [Exception('unexpected 1')]
        mock_now_side_effect = [0]
        self._exceptions.append(Exception('unexpected 1'))
        mock_now_side_effect.append(10)
        self._exceptions.append(Exception('unexpected 1'))
        mock_now_side_effect.extend([100, 100, 105])
        self._exceptions.append(Exception('unexpected 2'))
        mock_now_side_effect.extend([110, 111])
        mock_now.side_effect = mock_now_side_effect
        self.exception_generator()
        self.assertEqual([], self._exceptions)
        self.assertEqual(7, len(mock_now.mock_calls))
        self.assertEqual(3, len(mock_log.mock_calls))
        mock_log.assert_has_calls([mock.call('Unexpected exception occurred 1 time(s)... retrying.'), mock.call('Unexpected exception occurred 2 time(s)... retrying.'), mock.call('Unexpected exception occurred 1 time(s)... retrying.')])