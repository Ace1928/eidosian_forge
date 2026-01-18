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
@mock.patch.object(clusterutils, 'time')
def test_wait_for_clus_group_migr_timeout(self, mock_time, mock_get_gr_state, mock_is_migr_pending):
    exp_wait_iterations = 3
    mock_listener = mock.Mock()
    mock_time.time.side_effect = range(exp_wait_iterations + 2)
    timeout = 10
    state_info = dict(state=mock.sentinel.current_state, status_info=mock.sentinel.status_info)
    events = [dict(status_info=mock.sentinel.migr_queued), dict(state=mock.sentinel.pending_state), queue.Empty]
    mock_get_gr_state.return_value = state_info
    mock_is_migr_pending.return_value = True
    mock_listener.get.side_effect = events
    self.assertRaises(exceptions.ClusterGroupMigrationTimeOut, self._clusterutils._wait_for_cluster_group_migration, mock_listener, mock.sentinel.group_name, mock.sentinel.group_handle, mock.sentinel.expected_state, timeout=timeout)
    mock_get_gr_state.assert_called_once_with(mock.sentinel.group_handle)
    exp_wait_times = [timeout - elapsed - 1 for elapsed in range(exp_wait_iterations)]
    mock_listener.get.assert_has_calls([mock.call(wait_time) for wait_time in exp_wait_times])
    mock_is_migr_pending.assert_has_calls([mock.call(mock.sentinel.current_state, mock.sentinel.status_info, mock.sentinel.expected_state), mock.call(mock.sentinel.current_state, mock.sentinel.migr_queued, mock.sentinel.expected_state), mock.call(mock.sentinel.pending_state, mock.sentinel.migr_queued, mock.sentinel.expected_state)])