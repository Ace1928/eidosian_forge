from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def test_wait_io_completion(self):
    self._ioutils._wait_io_completion(mock.sentinel.event)
    self._mock_run.assert_called_once_with(ioutils.kernel32.WaitForSingleObjectEx, mock.sentinel.event, ioutils.WAIT_INFINITE_TIMEOUT, True, error_ret_vals=[w_const.WAIT_FAILED], **self._run_args)