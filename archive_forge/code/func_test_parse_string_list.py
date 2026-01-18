import collections
import ctypes
from unittest import mock
import ddt
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import iscsi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
def test_parse_string_list(self):
    self._mock_ctypes()
    fake_buff = 'fake\x00buff\x00\x00'
    self._ctypes.cast.return_value = fake_buff
    str_list = self._initiator._parse_string_list(fake_buff, len(fake_buff))
    self.assertEqual(['fake', 'buff'], str_list)
    self._ctypes.cast.assert_called_once_with(fake_buff, self._ctypes.POINTER.return_value)
    self._ctypes.POINTER.assert_called_once_with(self._ctypes.c_wchar)