import socket
import unittest
import httplib2
from six.moves import http_client
from mock import patch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
class HttpWrapperTest(unittest.TestCase):

    def testRequestBodyUsesLengthProperty(self):
        http_wrapper.Request(body=RaisesExceptionOnLen())

    def testRequestBodyWithLen(self):
        http_wrapper.Request(body='burrito')

    @unittest.skipIf(not _TOKEN_REFRESH_STATUS_AVAILABLE, 'oauth2client<1.5 lacks HttpAccessTokenRefreshError.')
    def testExceptionHandlerHttpAccessTokenError(self):
        exception_arg = HttpAccessTokenRefreshError(status=503)
        retry_args = http_wrapper.ExceptionRetryArgs(http={'connections': {}}, http_request=_MockHttpRequest(), exc=exception_arg, num_retries=0, max_retry_wait=0, total_wait_sec=0)
        with patch('time.sleep', return_value=None):
            http_wrapper.HandleExceptionsAndRebuildHttpConnections(retry_args)

    @unittest.skipIf(not _TOKEN_REFRESH_STATUS_AVAILABLE, 'oauth2client<1.5 lacks HttpAccessTokenRefreshError.')
    def testExceptionHandlerHttpAccessTokenErrorRaises(self):
        exception_arg = HttpAccessTokenRefreshError(status=200)
        retry_args = http_wrapper.ExceptionRetryArgs(http={'connections': {}}, http_request=_MockHttpRequest(), exc=exception_arg, num_retries=0, max_retry_wait=0, total_wait_sec=0)
        with self.assertRaises(HttpAccessTokenRefreshError):
            with patch('time.sleep', return_value=None):
                http_wrapper.HandleExceptionsAndRebuildHttpConnections(retry_args)

    def testExceptionHandlerAccessTokenErrorRaises(self):
        exception_arg = AccessTokenRefreshError()
        retry_args = http_wrapper.ExceptionRetryArgs(http={'connections': {}}, http_request=_MockHttpRequest(), exc=exception_arg, num_retries=0, max_retry_wait=0, total_wait_sec=0)
        with self.assertRaises(AccessTokenRefreshError):
            with patch('time.sleep', return_value=None):
                http_wrapper.HandleExceptionsAndRebuildHttpConnections(retry_args)

    def testDefaultExceptionHandler(self):
        """Ensures exception handles swallows (retries)"""
        mock_http_content = 'content'.encode('utf8')
        for exception_arg in (http_client.BadStatusLine('line'), http_client.IncompleteRead('partial'), http_client.ResponseNotReady(), socket.error(), socket.gaierror(), httplib2.ServerNotFoundError(), ValueError(), exceptions.RequestError(), exceptions.BadStatusCodeError({'status': 503}, mock_http_content, 'url'), exceptions.RetryAfterError({'status': 429}, mock_http_content, 'url', 0)):
            retry_args = http_wrapper.ExceptionRetryArgs(http={'connections': {}}, http_request=_MockHttpRequest(), exc=exception_arg, num_retries=0, max_retry_wait=0, total_wait_sec=0)
            with patch('time.sleep', return_value=None):
                http_wrapper.HandleExceptionsAndRebuildHttpConnections(retry_args)