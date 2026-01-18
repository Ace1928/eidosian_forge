import ctypes
from unittest import mock
import ddt
from six.moves import queue
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import clusterutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
@mock.patch('ctypes.byref')
def test_process_status_info_change_event(self, mock_byref):
    self._clusapi.get_cluster_group_status_info.return_value = mock.sentinel.status_info
    mock_byref.side_effect = lambda x: ('byref', x)
    notif_key = self._listener._NOTIF_KEY_GROUP_COMMON_PROP
    event = self._get_fake_event(notif_key=notif_key)
    exp_proc_evt = self._get_exp_processed_event(event, status_info=mock.sentinel.status_info)
    proc_evt = self._listener._process_event(event)
    self.assertEqual(exp_proc_evt, proc_evt)
    self._clusapi.get_cluster_group_status_info.assert_called_once_with(mock_byref(mock.sentinel.buff), mock.sentinel.buff_sz)