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
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_remove_persistent_login')
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_get_iscsi_persistent_logins')
def test_remove_target_persistent_logins(self, mock_get_iscsi_persistent_logins, mock_remove_persistent_login):
    fake_persistent_login = mock.Mock(TargetName=mock.sentinel.target_iqn)
    mock_get_iscsi_persistent_logins.return_value = [fake_persistent_login]
    self._initiator._remove_target_persistent_logins(mock.sentinel.target_iqn)
    mock_remove_persistent_login.assert_called_once_with(fake_persistent_login)
    mock_get_iscsi_persistent_logins.assert_called_once_with()