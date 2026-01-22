import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
class LogTracingTestCase(base.TestCase):
    """Test out the log tracing."""

    def test_utils_trace_method_default_logger(self):
        mock_log = self.mock_object(utils, 'LOG')

        @utils.trace
        def _trace_test_method_custom_logger(*args, **kwargs):
            return 'OK'
        result = _trace_test_method_custom_logger()
        self.assertEqual('OK', result)
        self.assertEqual(2, mock_log.debug.call_count)

    def test_utils_trace_method_inner_decorator(self):
        mock_logging = self.mock_object(utils, 'logging')
        mock_log = mock.Mock()
        mock_log.isEnabledFor = lambda x: True
        mock_logging.getLogger = mock.Mock(return_value=mock_log)

        def _test_decorator(f):

            def blah(*args, **kwargs):
                return f(*args, **kwargs)
            return blah

        @_test_decorator
        @utils.trace
        def _trace_test_method(*args, **kwargs):
            return 'OK'
        result = _trace_test_method(self)
        self.assertEqual('OK', result)
        self.assertEqual(2, mock_log.debug.call_count)
        for call in mock_log.debug.call_args_list:
            self.assertIn('_trace_test_method', str(call))
            self.assertNotIn('blah', str(call))

    def test_utils_trace_method_outer_decorator(self):
        mock_logging = self.mock_object(utils, 'logging')
        mock_log = mock.Mock()
        mock_log.isEnabledFor = lambda x: True
        mock_logging.getLogger = mock.Mock(return_value=mock_log)

        def _test_decorator(f):

            def blah(*args, **kwargs):
                return f(*args, **kwargs)
            return blah

        @utils.trace
        @_test_decorator
        def _trace_test_method(*args, **kwargs):
            return 'OK'
        result = _trace_test_method(self)
        self.assertEqual('OK', result)
        self.assertEqual(2, mock_log.debug.call_count)
        for call in mock_log.debug.call_args_list:
            self.assertNotIn('_trace_test_method', str(call))
            self.assertIn('blah', str(call))

    def test_utils_trace_method_outer_decorator_with_functools(self):
        mock_log = mock.Mock()
        mock_log.isEnabledFor = lambda x: True
        self.mock_object(utils.logging, 'getLogger', mock_log)
        mock_log = self.mock_object(utils, 'LOG')

        def _test_decorator(f):

            @functools.wraps(f)
            def wraps(*args, **kwargs):
                return f(*args, **kwargs)
            return wraps

        @utils.trace
        @_test_decorator
        def _trace_test_method(*args, **kwargs):
            return 'OK'
        result = _trace_test_method()
        self.assertEqual('OK', result)
        self.assertEqual(2, mock_log.debug.call_count)
        for call in mock_log.debug.call_args_list:
            self.assertIn('_trace_test_method', str(call))
            self.assertNotIn('wraps', str(call))

    def test_utils_trace_method_with_exception(self):
        self.LOG = self.mock_object(utils, 'LOG')

        @utils.trace
        def _trace_test_method(*args, **kwargs):
            raise exception.VolumeDeviceNotFound('test message')
        self.assertRaises(exception.VolumeDeviceNotFound, _trace_test_method)
        exception_log = self.LOG.debug.call_args_list[1]
        self.assertIn('exception', str(exception_log))
        self.assertIn('test message', str(exception_log))

    def test_utils_trace_method_with_time(self):
        mock_logging = self.mock_object(utils, 'logging')
        mock_log = mock.Mock()
        mock_log.isEnabledFor = lambda x: True
        mock_logging.getLogger = mock.Mock(return_value=mock_log)
        mock_time = mock.Mock(side_effect=[3.1, 6])
        self.mock_object(time, 'time', mock_time)

        @utils.trace
        def _trace_test_method(*args, **kwargs):
            return 'OK'
        result = _trace_test_method(self)
        self.assertEqual('OK', result)
        return_log = mock_log.debug.call_args_list[1]
        self.assertIn('2900', str(return_log))

    def test_utils_trace_method_with_password_dict(self):
        mock_logging = self.mock_object(utils, 'logging')
        mock_log = mock.Mock()
        mock_log.isEnabledFor = lambda x: True
        mock_logging.getLogger = mock.Mock(return_value=mock_log)

        @utils.trace
        def _trace_test_method(*args, **kwargs):
            return {'something': 'test', 'password': 'Now you see me'}
        result = _trace_test_method(self)
        expected_unmasked_dict = {'something': 'test', 'password': 'Now you see me'}
        self.assertEqual(expected_unmasked_dict, result)
        self.assertEqual(2, mock_log.debug.call_count)
        self.assertIn("'password': '***'", str(mock_log.debug.call_args_list[1]))

    def test_utils_trace_method_with_password_str(self):
        mock_logging = self.mock_object(utils, 'logging')
        mock_log = mock.Mock()
        mock_log.isEnabledFor = lambda x: True
        mock_logging.getLogger = mock.Mock(return_value=mock_log)

        @utils.trace
        def _trace_test_method(*args, **kwargs):
            return "'adminPass': 'Now you see me'"
        result = _trace_test_method(self)
        expected_unmasked_str = "'adminPass': 'Now you see me'"
        self.assertEqual(expected_unmasked_str, result)
        self.assertEqual(2, mock_log.debug.call_count)
        self.assertIn("'adminPass': '***'", str(mock_log.debug.call_args_list[1]))

    def test_utils_trace_method_with_password_in_formal_params(self):
        mock_logging = self.mock_object(utils, 'logging')
        mock_log = mock.Mock()
        mock_log.isEnabledFor = lambda x: True
        mock_logging.getLogger = mock.Mock(return_value=mock_log)

        @utils.trace
        def _trace_test_method(*args, **kwargs):
            self.assertEqual('verybadpass', kwargs['connection']['data']['auth_password'])
            pass
        connector_properties = {'data': {'auth_password': 'verybadpass'}}
        _trace_test_method(self, connection=connector_properties)
        self.assertEqual(2, mock_log.debug.call_count)
        self.assertIn("'auth_password': '***'", str(mock_log.debug.call_args_list[0]))