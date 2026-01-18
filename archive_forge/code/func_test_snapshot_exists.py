from unittest import mock
import ddt
from oslo_utils import units
from oslo_vmware.objects import datastore
from oslo_vmware import vim_util
from os_brick import exception
from os_brick.initiator.connectors import vmware
from os_brick.tests.initiator import test_connector
@ddt.data((None, False), ([mock.sentinel.snap], True))
@ddt.unpack
def test_snapshot_exists(self, snap_list, exp_return_value):
    snapshot = mock.Mock(rootSnapshotList=snap_list)
    session = mock.Mock()
    session.invoke_api.return_value = snapshot
    backing = mock.sentinel.backing
    ret = self._connector._snapshot_exists(session, backing)
    self.assertEqual(exp_return_value, ret)
    session.invoke_api.assert_called_once_with(vim_util, 'get_object_property', session.vim, backing, 'snapshot')