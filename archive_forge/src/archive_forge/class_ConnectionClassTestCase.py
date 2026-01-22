import os
import ssl
import sys
import socket
from unittest import mock
from unittest.mock import Mock, patch
import requests_mock
from requests.exceptions import ConnectTimeout
import libcloud.common.base
from libcloud.http import LibcloudConnection, SignedHTTPSAdapter, LibcloudBaseConnection
from libcloud.test import unittest, no_internet
from libcloud.utils.py3 import assertRaisesRegex
from libcloud.common.base import Response, Connection, CertificateConnection
from libcloud.utils.retry import RETRY_EXCEPTIONS, Retry, RetryForeverOnRateLimitError
from libcloud.common.exceptions import RateLimitReachedError
class ConnectionClassTestCase(unittest.TestCase):

    def setUp(self):
        self.originalConnect = Connection.connect
        self.originalResponseCls = Connection.responseCls
        Connection.connect = Mock()
        Connection.responseCls = Mock()
        Connection.allow_insecure = True

    def tearDown(self):
        Connection.connect = self.originalConnect
        Connection.responseCls = Connection.responseCls
        Connection.allow_insecure = True

    def test_dont_allow_insecure(self):
        Connection.allow_insecure = True
        Connection(secure=False)
        Connection.allow_insecure = False
        expected_msg = 'Non https connections are not allowed \\(use secure=True\\)'
        assertRaisesRegex(self, ValueError, expected_msg, Connection, secure=False)

    def test_cache_busting(self):
        params1 = {'foo1': 'bar1', 'foo2': 'bar2'}
        params2 = [('foo1', 'bar1'), ('foo2', 'bar2')]
        con = Connection()
        con.connection = Mock()
        con.pre_connect_hook = Mock()
        con.pre_connect_hook.return_value = ({}, {})
        con.cache_busting = False
        con.request(action='/path', params=params1)
        args, kwargs = con.pre_connect_hook.call_args
        self.assertFalse('cache-busting' in args[0])
        self.assertEqual(args[0], params1)
        con.request(action='/path', params=params2)
        args, kwargs = con.pre_connect_hook.call_args
        self.assertFalse('cache-busting' in args[0])
        self.assertEqual(args[0], params2)
        con.cache_busting = True
        con.request(action='/path', params=params1)
        args, kwargs = con.pre_connect_hook.call_args
        self.assertTrue('cache-busting' in args[0])
        con.request(action='/path', params=params2)
        args, kwargs = con.pre_connect_hook.call_args
        self.assertTrue('cache-busting' in args[0][len(params2)])

    def test_context_is_reset_after_request_has_finished(self):
        context = {'foo': 'bar'}

        def responseCls(connection, response) -> mock.MagicMock:
            connection.called = True
            self.assertEqual(connection.context, context)
            return mock.MagicMock(spec=Response)
        con = Connection()
        con.called = False
        con.connection = Mock()
        con.responseCls = responseCls
        con.set_context(context)
        self.assertEqual(con.context, context)
        con.request('/')
        self.assertTrue(con.called)
        self.assertEqual(con.context, {})
        con = Connection(timeout=1, retry_delay=0.1)
        con.connection = Mock()
        con.set_context(context)
        self.assertEqual(con.context, context)
        con.connection.request = Mock(side_effect=ssl.SSLError())
        try:
            con.request('/')
        except ssl.SSLError:
            pass
        self.assertEqual(con.context, {})
        con.connection = Mock()
        con.set_context(context)
        self.assertEqual(con.context, context)
        con.responseCls = Mock(side_effect=ValueError())
        try:
            con.request('/')
        except ValueError:
            pass
        self.assertEqual(con.context, {})

    def _raise_socket_error(self):
        raise socket.gaierror('')

    @patch('libcloud.common.base.Connection.request')
    def test_retry_with_sleep(self, mock_connect):
        con = Connection()
        con.connection = Mock()
        mock_connect.side_effect = socket.gaierror('')
        retry_request = Retry(timeout=1, retry_delay=0.1, backoff=1)
        self.assertRaises(socket.gaierror, retry_request(con.request), action='/')
        self.assertGreater(mock_connect.call_count, 1, 'Retry logic failed')

    @patch('libcloud.common.base.Connection.request')
    def test_retry_with_timeout(self, mock_connect):
        con = Connection()
        con.connection = Mock()
        mock_connect.side_effect = socket.gaierror('')
        retry_request = Retry(timeout=1, retry_delay=0.1, backoff=1)
        self.assertRaises(socket.gaierror, retry_request(con.request), action='/')
        self.assertGreater(mock_connect.call_count, 1, 'Retry logic failed')

    @patch('libcloud.common.base.Connection.request')
    def test_retry_with_backoff(self, mock_connect):
        con = Connection()
        con.connection = Mock()
        mock_connect.side_effect = socket.gaierror('')
        retry_request = Retry(timeout=1, retry_delay=0.1, backoff=1)
        self.assertRaises(socket.gaierror, retry_request(con.request), action='/')
        self.assertGreater(mock_connect.call_count, 1, 'Retry logic failed')

    @patch('libcloud.common.base.Connection.request')
    def test_retry_rate_limit_error_timeout(self, mock_connect):
        con = Connection()
        con.connection = Mock()
        mock_connect.__name__ = 'mock_connect'
        headers = {'retry-after': 0.2}
        mock_connect.side_effect = RateLimitReachedError(headers=headers)
        retry_request = Retry(timeout=1, retry_delay=0.1, backoff=1)
        self.assertRaises(RateLimitReachedError, retry_request(con.request), action='/')
        self.assertGreater(mock_connect.call_count, 1, 'Retry logic failed')

    @patch('libcloud.common.base.Connection.request')
    def test_retry_rate_limit_error_forever_with_old_retry_class(self, mock_connect):
        con = Connection()
        con.connection = Mock()
        self.retry_counter = 0

        def mock_connect_side_effect(*args, **kwargs):
            self.retry_counter += 1
            if self.retry_counter < 4:
                headers = {'retry-after': 0.1}
                raise RateLimitReachedError(headers=headers)
            return 'success'
        mock_connect.__name__ = 'mock_connect'
        mock_connect.side_effect = mock_connect_side_effect
        retry_request = RetryForeverOnRateLimitError(timeout=1, retry_delay=0.1, backoff=1)
        retry_request(con.request)(action='/')
        result = retry_request(con.request)(action='/')
        self.assertEqual(result, 'success')
        self.assertEqual(mock_connect.call_count, 5, 'Retry logic failed')

    @patch('libcloud.common.base.Connection.request')
    def test_retry_should_not_retry_on_non_defined_exception(self, mock_connect):
        con = Connection()
        con.connection = Mock()
        self.retry_counter = 0
        mock_connect.__name__ = 'mock_connect'
        mock_connect.side_effect = ValueError('should not retry this error')
        retry_request = Retry(timeout=5, retry_delay=0.1, backoff=1)
        self.assertRaisesRegex(ValueError, 'should not retry this error', retry_request(con.request), action='/')
        self.assertEqual(mock_connect.call_count, 1, 'Retry logic failed')

    @patch('libcloud.common.base.Connection.request')
    def test_retry_rate_limit_error_success_on_second_attempt(self, mock_connect):
        con = Connection()
        con.connection = Mock()
        self.retry_counter = 0

        def mock_connect_side_effect(*args, **kwargs):
            self.retry_counter += 1
            if self.retry_counter < 2:
                headers = {'retry-after': 0.2}
                raise RateLimitReachedError(headers=headers)
            return 'success'
        mock_connect.__name__ = 'mock_connect'
        mock_connect.side_effect = mock_connect_side_effect
        retry_request = Retry(timeout=1, retry_delay=0.1, backoff=1)
        result = retry_request(con.request)(action='/')
        self.assertEqual(result, 'success')
        self.assertEqual(mock_connect.call_count, 2, 'Retry logic failed')

    @patch('libcloud.common.base.Connection.request')
    def test_retry_on_all_default_retry_exception_classes(self, mock_connect):
        con = Connection()
        con.connection = Mock()
        self.retry_counter = 0

        def mock_connect_side_effect(*args, **kwargs):
            self.retry_counter += 1
            if self.retry_counter < len(RETRY_EXCEPTIONS):
                raise RETRY_EXCEPTIONS[self.retry_counter]
            return 'success'
        mock_connect.__name__ = 'mock_connect'
        mock_connect.side_effect = mock_connect_side_effect
        retry_request = Retry(timeout=1, retry_delay=0.1, backoff=1)
        result = retry_request(con.request)(action='/')
        self.assertEqual(result, 'success')
        self.assertEqual(mock_connect.call_count, len(RETRY_EXCEPTIONS), 'Retry logic failed')

    def test_request_parses_errors(self):

        class ThrowingResponse(Response):

            def __init__(self, *_, **__):
                super().__init__(mock.MagicMock(), mock.MagicMock())

            def parse_body(self):
                return super().parse_body()

            def parse_error(self):
                raise RateLimitReachedError()

            def success(self):
                return False
        con = Connection()
        con.connection = Mock()
        con.responseCls = ThrowingResponse
        with self.assertRaises(RateLimitReachedError):
            con.request(action='/')

    def test_parse_errors_can_be_retried(self):

        class RetryableThrowingError(Response):
            parse_error_counter: int = 0
            success_counter: int = 0

            def __init__(self, *_, **__):
                super().__init__(mock.MagicMock(), mock.MagicMock())

            def parse_body(self):
                return super().parse_body()

            def parse_error(self):
                RetryableThrowingError.parse_error_counter += 1
                if RetryableThrowingError.parse_error_counter > 1:
                    return 'success'
                else:
                    raise RateLimitReachedError()

            def success(self):
                RetryableThrowingError.success_counter += 1
                if RetryableThrowingError.success_counter > 1:
                    return True
                else:
                    return False
        con = Connection()
        con.connection = Mock()
        con.responseCls = RetryableThrowingError
        result = con.request(action='/', retry_failed=True)
        self.assertEqual(result.success(), True)