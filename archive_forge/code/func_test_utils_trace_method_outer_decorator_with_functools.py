import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
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