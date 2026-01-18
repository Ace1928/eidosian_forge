import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@mock.patch('tenacity.nap.sleep')
def test_retry_exit_code(self, sleep_mock):
    exit_code = 5
    exception = utils.processutils.ProcessExecutionError

    @utils.retry(retry=utils.retry_if_exit_code, retry_param=exit_code)
    def raise_retriable_exit_code():
        raise exception(exit_code=exit_code)
    self.assertRaises(exception, raise_retriable_exit_code)
    self.assertEqual(0, sleep_mock.call_count)