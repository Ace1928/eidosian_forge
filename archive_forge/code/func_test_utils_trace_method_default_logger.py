import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
def test_utils_trace_method_default_logger(self):
    mock_log = self.mock_object(utils, 'LOG')

    @utils.trace
    def _trace_test_method_custom_logger(*args, **kwargs):
        return 'OK'
    result = _trace_test_method_custom_logger()
    self.assertEqual('OK', result)
    self.assertEqual(2, mock_log.debug.call_count)