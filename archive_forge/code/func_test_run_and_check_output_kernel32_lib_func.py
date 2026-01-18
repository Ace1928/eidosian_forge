from unittest import mock
import ddt
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def test_run_and_check_output_kernel32_lib_func(self):
    ret_val = 0
    self._test_run_and_check_output(ret_val=ret_val, expected_exc=exceptions.Win32Exception, kernel32_lib_func=True)