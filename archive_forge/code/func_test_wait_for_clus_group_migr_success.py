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
@mock.patch.object(clusterutils.ClusterUtils, '_is_migration_pending')
@mock.patch.object(clusterutils.ClusterUtils, '_get_cluster_group_state')
def test_wait_for_clus_group_migr_success(self, mock_get_gr_state, mock_is_migr_pending):
    mock_listener = mock.Mock()
    state_info = dict(state=mock.sentinel.current_state, status_info=mock.sentinel.status_info)
    mock_get_gr_state.return_value = state_info
    mock_is_migr_pending.side_effect = [True, False]
    mock_listener.get.return_value = {}
    self._clusterutils._wait_for_cluster_group_migration(mock_listener, mock.sentinel.group_name, mock.sentinel.group_handle, mock.sentinel.expected_state, timeout=None)
    mock_listener.get.assert_called_once_with(None)