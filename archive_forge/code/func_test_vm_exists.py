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
@ddt.data(True, False)
def test_vm_exists(self, exists):
    self._cmgr.open_cluster_resource.side_effect = None if exists else exceptions.ClusterObjectNotFound('test')
    self.assertEqual(exists, self._clusterutils.vm_exists(self._FAKE_VM_NAME))
    self._cmgr.open_cluster_resource.assert_called_once_with(self._FAKE_RESOURCEGROUP_NAME)