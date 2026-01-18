import builtins
import errno
import os.path
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import executor
from os_brick.initiator.connectors import nvmeof
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base as test_base
from os_brick.tests.initiator import test_connector
from os_brick import utils
@ddt.data('traddr=portal2,trsvcid=port2', 'traddr=portal2,trsvcid=port2,src_addr=myip')
@mock.patch.object(nvmeof, 'sysfs_property')
@mock.patch('glob.glob')
def test_set_portals_controllers(self, addr, mock_glob, mock_sysfs):
    """Look in sysfs for the device paths."""
    portal = nvmeof.Portal(self.target, 'portal4', 'port4', 'tcp')
    portal.controller = 'nvme0'
    self.target.portals.insert(0, portal)
    self.target.portals.append(nvmeof.Portal(self.target, 'portal5', 'port5', 'tcp'))
    self.target.host_nqn = 'nqn'
    mock_glob.return_value = ['/sys/class/nvme-fabrics/ctl/nvme0', '/sys/class/nvme-fabrics/ctl/nvme1', '/sys/class/nvme-fabrics/ctl/nvme2', '/sys/class/nvme-fabrics/ctl/nvme3', '/sys/class/nvme-fabrics/ctl/nvme4', '/sys/class/nvme-fabrics/ctl/nvme5']
    mock_sysfs.side_effect = ['wrong-nqn', self.target.nqn, 'rdma', 'traddr=portal5,trsvcid=port5', 'nqn', self.target.nqn, 'rdma', 'traddr=portal2,trsvcid=port2', 'badnqn', self.target.nqn, 'tcp', addr, 'nqn', self.target.nqn, 'tcp', 'traddr=portal5,trsvcid=port5', None]
    self.target.set_portals_controllers()
    mock_glob.assert_called_once_with('/sys/class/nvme-fabrics/ctl/nvme*')
    expected_calls = [mock.call('subsysnqn', '/sys/class/nvme-fabrics/ctl/nvme1'), mock.call('subsysnqn', '/sys/class/nvme-fabrics/ctl/nvme2'), mock.call('transport', '/sys/class/nvme-fabrics/ctl/nvme2'), mock.call('address', '/sys/class/nvme-fabrics/ctl/nvme2'), mock.call('hostnqn', '/sys/class/nvme-fabrics/ctl/nvme2'), mock.call('subsysnqn', '/sys/class/nvme-fabrics/ctl/nvme3'), mock.call('transport', '/sys/class/nvme-fabrics/ctl/nvme3'), mock.call('address', '/sys/class/nvme-fabrics/ctl/nvme3'), mock.call('hostnqn', '/sys/class/nvme-fabrics/ctl/nvme3'), mock.call('subsysnqn', '/sys/class/nvme-fabrics/ctl/nvme4'), mock.call('transport', '/sys/class/nvme-fabrics/ctl/nvme4'), mock.call('address', '/sys/class/nvme-fabrics/ctl/nvme4'), mock.call('hostnqn', '/sys/class/nvme-fabrics/ctl/nvme4'), mock.call('subsysnqn', '/sys/class/nvme-fabrics/ctl/nvme5'), mock.call('transport', '/sys/class/nvme-fabrics/ctl/nvme5'), mock.call('address', '/sys/class/nvme-fabrics/ctl/nvme5'), mock.call('hostnqn', '/sys/class/nvme-fabrics/ctl/nvme5')]
    self.assertEqual(len(expected_calls), mock_sysfs.call_count)
    mock_sysfs.assert_has_calls(expected_calls)
    self.assertEqual('nvme0', self.target.portals[0].controller)
    self.assertIsNone(self.target.portals[1].controller)
    self.assertEqual('nvme4', self.target.portals[2].controller)
    self.assertEqual('nvme5', self.target.portals[3].controller)