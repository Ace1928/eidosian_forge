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
def test_process_event_dropped(self, mock_byref):
    event = self._get_fake_event(cluster_object_name='other_group_name')
    self.assertIsNone(self._listener._process_event(event))
    event = self._get_fake_event(notif_key=2)
    self.assertIsNone(self._listener._process_event(event))
    notif_key = self._listener._NOTIF_KEY_GROUP_COMMON_PROP
    self._clusapi.get_cluster_group_status_info.side_effect = exceptions.ClusterPropertyListEntryNotFound(property_name='fake_prop_name')
    event = self._get_fake_event(notif_key=notif_key)
    self.assertIsNone(self._listener._process_event(event))