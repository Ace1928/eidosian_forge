from unittest import mock
import ddt
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def test_run_and_check_output_ignored_error(self):
    ret_val = 1
    ignored_err_codes = [ret_val]
    self._test_run_and_check_output(ret_val=ret_val, ignored_error_codes=ignored_err_codes)