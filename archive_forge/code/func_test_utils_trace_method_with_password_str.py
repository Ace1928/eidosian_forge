import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
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