import ctypes
from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import _clusapi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def test_open_cluster_enum(self):
    with self._cmgr.open_cluster_enum(mock.sentinel.object_type) as f:
        self._clusapi_utils.open_cluster.assert_called_once_with(None)
        self._clusapi_utils.open_cluster_enum.assert_called_once_with(self._clusapi_utils.open_cluster.return_value, mock.sentinel.object_type)
        self.assertEqual(f, self._clusapi_utils.open_cluster_enum.return_value)
    self._clusapi_utils.close_cluster_enum.assert_called_once_with(self._clusapi_utils.open_cluster_enum.return_value)
    self._clusapi_utils.close_cluster.assert_called_once_with(self._clusapi_utils.open_cluster.return_value)