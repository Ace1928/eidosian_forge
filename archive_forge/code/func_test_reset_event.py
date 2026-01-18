from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def test_reset_event(self):
    self._ioutils._reset_event(mock.sentinel.event)
    self._mock_run.assert_called_once_with(ioutils.kernel32.ResetEvent, mock.sentinel.event, **self._run_args)