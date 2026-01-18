from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def test_wait_named_pipe(self):
    fake_timeout_s = 10
    self._ioutils.wait_named_pipe(mock.sentinel.pipe_name, timeout=fake_timeout_s)
    self._mock_run.assert_called_once_with(ioutils.kernel32.WaitNamedPipeW, self._ctypes.c_wchar_p(mock.sentinel.pipe_name), fake_timeout_s * 1000, **self._run_args)