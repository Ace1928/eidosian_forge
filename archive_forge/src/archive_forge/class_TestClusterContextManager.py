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
@ddt.ddt
class TestClusterContextManager(test_base.OsWinBaseTestCase):
    _autospec_classes = [_clusapi_utils.ClusApiUtils]

    def setUp(self):
        super(TestClusterContextManager, self).setUp()
        self._cmgr = _clusapi_utils.ClusterContextManager()
        self._clusapi_utils = self._cmgr._clusapi_utils

    @ddt.data(None, mock.sentinel.cluster_name)
    def test_open_cluster(self, cluster_name):
        with self._cmgr.open_cluster(cluster_name) as f:
            self._clusapi_utils.open_cluster.assert_called_once_with(cluster_name)
            self.assertEqual(f, self._clusapi_utils.open_cluster.return_value)
        self._clusapi_utils.close_cluster.assert_called_once_with(self._clusapi_utils.open_cluster.return_value)

    def test_open_cluster_group(self):
        with self._cmgr.open_cluster_group(mock.sentinel.group_name) as f:
            self._clusapi_utils.open_cluster.assert_called_once_with(None)
            self._clusapi_utils.open_cluster_group.assert_called_once_with(self._clusapi_utils.open_cluster.return_value, mock.sentinel.group_name)
            self.assertEqual(f, self._clusapi_utils.open_cluster_group.return_value)
        self._clusapi_utils.close_cluster_group.assert_called_once_with(self._clusapi_utils.open_cluster_group.return_value)
        self._clusapi_utils.close_cluster.assert_called_once_with(self._clusapi_utils.open_cluster.return_value)

    def test_open_missing_cluster_group(self):
        exc = exceptions.ClusterWin32Exception(func_name='OpenClusterGroup', message='expected exception', error_code=w_const.ERROR_GROUP_NOT_FOUND)
        self._clusapi_utils.open_cluster_group.side_effect = exc
        self.assertRaises(exceptions.ClusterObjectNotFound, self._cmgr.open_cluster_group(mock.sentinel.group_name).__enter__)

    def test_open_cluster_group_with_handle(self):
        with self._cmgr.open_cluster_group(mock.sentinel.group_name, cluster_handle=mock.sentinel.cluster_handle) as f:
            self._clusapi_utils.open_cluster.assert_not_called()
            self._clusapi_utils.open_cluster_group.assert_called_once_with(mock.sentinel.cluster_handle, mock.sentinel.group_name)
            self.assertEqual(f, self._clusapi_utils.open_cluster_group.return_value)
        self._clusapi_utils.close_cluster_group.assert_called_once_with(self._clusapi_utils.open_cluster_group.return_value)
        self._clusapi_utils.close_cluster.assert_not_called()

    def test_open_cluster_resource(self):
        with self._cmgr.open_cluster_resource(mock.sentinel.res_name) as f:
            self._clusapi_utils.open_cluster.assert_called_once_with(None)
            self._clusapi_utils.open_cluster_resource.assert_called_once_with(self._clusapi_utils.open_cluster.return_value, mock.sentinel.res_name)
            self.assertEqual(f, self._clusapi_utils.open_cluster_resource.return_value)
        self._clusapi_utils.close_cluster_resource.assert_called_once_with(self._clusapi_utils.open_cluster_resource.return_value)
        self._clusapi_utils.close_cluster.assert_called_once_with(self._clusapi_utils.open_cluster.return_value)

    def test_open_cluster_node(self):
        with self._cmgr.open_cluster_node(mock.sentinel.node_name) as f:
            self._clusapi_utils.open_cluster.assert_called_once_with(None)
            self._clusapi_utils.open_cluster_node.assert_called_once_with(self._clusapi_utils.open_cluster.return_value, mock.sentinel.node_name)
            self.assertEqual(f, self._clusapi_utils.open_cluster_node.return_value)
        self._clusapi_utils.close_cluster_node.assert_called_once_with(self._clusapi_utils.open_cluster_node.return_value)
        self._clusapi_utils.close_cluster.assert_called_once_with(self._clusapi_utils.open_cluster.return_value)

    def test_open_cluster_enum(self):
        with self._cmgr.open_cluster_enum(mock.sentinel.object_type) as f:
            self._clusapi_utils.open_cluster.assert_called_once_with(None)
            self._clusapi_utils.open_cluster_enum.assert_called_once_with(self._clusapi_utils.open_cluster.return_value, mock.sentinel.object_type)
            self.assertEqual(f, self._clusapi_utils.open_cluster_enum.return_value)
        self._clusapi_utils.close_cluster_enum.assert_called_once_with(self._clusapi_utils.open_cluster_enum.return_value)
        self._clusapi_utils.close_cluster.assert_called_once_with(self._clusapi_utils.open_cluster.return_value)

    def test_invalid_handle_type(self):
        self.assertRaises(exceptions.Invalid, self._cmgr._open(handle_type=None).__enter__)
        self.assertRaises(exceptions.Invalid, self._cmgr._close, mock.sentinel.handle, handle_type=None)