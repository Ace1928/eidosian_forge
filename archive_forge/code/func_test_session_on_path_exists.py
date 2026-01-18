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
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_get_iscsi_session_devices')
def test_session_on_path_exists(self, mock_get_iscsi_session_devices):
    mock_device = mock.Mock(InitiatorName=mock.sentinel.initiator_name)
    mock_get_iscsi_session_devices.return_value = [mock_device]
    fake_connection = mock.Mock(TargetAddress=mock.sentinel.portal_addr, TargetSocket=mock.sentinel.portal_port)
    fake_connections = [mock.Mock(), fake_connection]
    fake_session = mock.Mock(ConnectionCount=len(fake_connections), Connections=fake_connections)
    fake_sessions = [mock.Mock(Connections=[], ConnectionCount=0), fake_session]
    session_on_path_exists = self._initiator._session_on_path_exists(fake_sessions, mock.sentinel.portal_addr, mock.sentinel.portal_port, mock.sentinel.initiator_name)
    self.assertTrue(session_on_path_exists)
    mock_get_iscsi_session_devices.assert_has_calls([mock.call(session.SessionId) for session in fake_sessions])