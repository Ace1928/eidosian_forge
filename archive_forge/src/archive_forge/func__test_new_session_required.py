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
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_get_iscsi_target_sessions')
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_session_on_path_exists')
def _test_new_session_required(self, mock_session_on_path_exists, mock_get_iscsi_target_sessions, sessions=None, mpio_enabled=False, session_on_path_exists=False):
    mock_get_iscsi_target_sessions.return_value = sessions
    mock_session_on_path_exists.return_value = session_on_path_exists
    expected_result = not sessions or (mpio_enabled and (not session_on_path_exists))
    result = self._initiator._new_session_required(mock.sentinel.target_iqn, mock.sentinel.portal_addr, mock.sentinel.portal_port, mock.sentinel.initiator_name, mpio_enabled)
    self.assertEqual(expected_result, result)
    if sessions and mpio_enabled:
        mock_session_on_path_exists.assert_called_once_with(sessions, mock.sentinel.portal_addr, mock.sentinel.portal_port, mock.sentinel.initiator_name)