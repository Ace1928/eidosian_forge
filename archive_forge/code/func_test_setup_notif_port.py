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
@mock.patch.object(clusterutils._ClusterEventListener, '_add_filter')
@mock.patch.object(clusapi_def, 'NOTIFY_FILTER_AND_TYPE')
def test_setup_notif_port(self, mock_filter_struct_cls, mock_add_filter):
    notif_filter = dict(object_type=mock.sentinel.object_type, filter_flags=mock.sentinel.filter_flags, notif_key=mock.sentinel.notif_key)
    self._listener._notif_filters_list = [notif_filter]
    self._listener._setup_notif_port()
    mock_filter_struct_cls.assert_called_once_with(dwObjectType=mock.sentinel.object_type, FilterFlags=mock.sentinel.filter_flags)
    mock_add_filter.assert_called_once_with(mock_filter_struct_cls.return_value, mock.sentinel.notif_key)