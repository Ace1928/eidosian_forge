import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
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