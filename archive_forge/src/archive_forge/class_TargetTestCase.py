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
@ddt.ddt
class TargetTestCase(test_base.TestCase):

    def setUp(self):
        self.conn_props_dict = {'target_nqn': 'nqn_value', 'vol_uuid': VOL_UUID, 'portals': [('portal1', 'port1', 'RoCEv2'), ('portal2', 'port2', 'anything')]}
        self.conn_props = nvmeof.NVMeOFConnProps(self.conn_props_dict)
        self.target = self.conn_props.targets[0]
        super().setUp()

    @mock.patch.object(nvmeof.Target, '__init__', return_value=None)
    def test_factory(self, mock_init):
        """Test Target factory

        The factory's parameter names take after the keys in the connection

        properties, and then calls the class init method that uses different
        names.
        """
        res = nvmeof.Target.factory(self.conn_props, **self.conn_props_dict)
        mock_init.assert_called_once_with(self.conn_props, self.conn_props_dict['target_nqn'], self.conn_props_dict['portals'], self.conn_props_dict['vol_uuid'], None, None, None, False)
        self.assertIsInstance(res, nvmeof.Target)

    @ddt.data(True, False)
    @mock.patch.object(nvmeof.Target, 'set_portals_controllers')
    @mock.patch.object(nvmeof.Portal, '__init__', return_value=None)
    def test_init(self, find_controllers, mock_init, mock_set_ctrls):
        """Init instantiates portals and may call set_portals_controllers."""
        target = nvmeof.Target(self.conn_props, 'nqn', self.conn_props_dict['portals'], VOL_UUID_NO_HYPHENS, NGUID_NO_HYPHENS, 'ns_id', 'host_nqn', find_controllers)
        self.assertEqual(self.conn_props, target.source_conn_props)
        self.assertEqual('nqn', target.nqn)
        self.assertEqual(VOL_UUID, target.uuid)
        self.assertEqual(NGUID, target.nguid)
        self.assertEqual('ns_id', target.ns_id)
        self.assertEqual('host_nqn', target.host_nqn)
        self.assertIsInstance(target.portals[0], nvmeof.Portal)
        self.assertIsInstance(target.portals[1], nvmeof.Portal)
        if find_controllers:
            mock_set_ctrls.assert_called_once_with()
        else:
            mock_set_ctrls.assert_not_called()
        self.assertEqual(2, mock_init.call_count)
        mock_init.assert_has_calls([mock.call(target, 'portal1', 'port1', 'RoCEv2'), mock.call(target, 'portal2', 'port2', 'anything')])

    @mock.patch.object(nvmeof.Target, '_get_nvme_devices')
    @mock.patch.object(nvmeof.Target, 'set_portals_controllers')
    @mock.patch.object(nvmeof.Portal, '__init__', return_value=None)
    def test_init_no_id(self, mock_init, mock_set_ctrls, mock_get_devs):
        """With no ID parameters query existing nvme devices."""
        target = nvmeof.Target(self.conn_props, 'nqn', self.conn_props_dict['portals'])
        self.assertEqual(self.conn_props, target.source_conn_props)
        self.assertEqual('nqn', target.nqn)
        for name in ('uuid', 'nguid', 'ns_id'):
            self.assertIsNone(getattr(target, name))
        self.assertIsInstance(target.portals[0], nvmeof.Portal)
        self.assertIsInstance(target.portals[1], nvmeof.Portal)
        mock_set_ctrls.assert_not_called()
        mock_get_devs.assert_called_once_with()
        self.assertEqual(2, mock_init.call_count)
        mock_init.assert_has_calls([mock.call(target, 'portal1', 'port1', 'RoCEv2'), mock.call(target, 'portal2', 'port2', 'anything')])

    @mock.patch('glob.glob', return_value=['/dev/nvme0n1', '/dev/nvme1n1'])
    def test__get_nvme_devices(self, mock_glob):
        """Test getting all nvme devices present in system."""
        res = self.target._get_nvme_devices()
        self.assertEqual(mock_glob.return_value, res)
        mock_glob.assert_called_once_with('/dev/nvme*n*')

    @mock.patch.object(nvmeof.Portal, 'is_live', new_callable=mock.PropertyMock)
    def test_live_portals(self, mock_is_live):
        """List with only live portals should be returned."""
        mock_is_live.side_effect = (True, False)
        res = self.target.live_portals
        self.assertListEqual([self.target.portals[0]], res)

    @mock.patch.object(nvmeof.Portal, 'state', new_callable=mock.PropertyMock)
    def test_present_portals(self, mock_state):
        """List with only live portals should be returned."""
        self.target.portals.extend(self.target.portals)
        mock_state.side_effect = (None, 'live', 'connecting', 'live')
        res = self.target.present_portals
        self.assertListEqual(self.target.portals[1:], res)

    @mock.patch('glob.glob')
    def test_set_portals_controllers_do_nothing(self, mock_glob):
        """Do nothing if all protals already have the controller name."""
        self.target.portals[0].controller = 'nvme0'
        self.target.portals[1].controller = 'nvme1'
        self.target.set_portals_controllers()
        mock_glob.assert_not_called()

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

    @mock.patch('os_brick.utils.get_host_nqn', mock.Mock(return_value='nqn'))
    @mock.patch.object(nvmeof, 'sysfs_property')
    @mock.patch('glob.glob')
    def test_set_portals_controllers_short_circuit(self, mock_glob, mock_sysfs):
        """Stops looking once we have found names for all portals."""
        self.target.portals[0].controller = 'nvme0'
        mock_glob.return_value = ['/sys/class/nvme-fabrics/ctl/nvme0', '/sys/class/nvme-fabrics/ctl/nvme1', '/sys/class/nvme-fabrics/ctl/nvme2', '/sys/class/nvme-fabrics/ctl/nvme3']
        mock_sysfs.side_effect = [self.target.nqn, 'tcp', 'traddr=portal2,trsvcid=port2', 'nqn']
        self.target.set_portals_controllers()
        mock_glob.assert_called_once_with('/sys/class/nvme-fabrics/ctl/nvme*')
        expected_calls = [mock.call('subsysnqn', '/sys/class/nvme-fabrics/ctl/nvme1'), mock.call('transport', '/sys/class/nvme-fabrics/ctl/nvme1'), mock.call('address', '/sys/class/nvme-fabrics/ctl/nvme1'), mock.call('hostnqn', '/sys/class/nvme-fabrics/ctl/nvme1')]
        self.assertEqual(len(expected_calls), mock_sysfs.call_count)
        mock_sysfs.assert_has_calls(expected_calls)
        self.assertEqual('nvme0', self.target.portals[0].controller)
        self.assertEqual('nvme1', self.target.portals[1].controller)

    @mock.patch.object(nvmeof.Target, 'present_portals', new_callable=mock.PropertyMock)
    @mock.patch.object(nvmeof.Target, 'live_portals', new_callable=mock.PropertyMock)
    def test_get_devices_first_live(self, mock_live, mock_present):
        """Return on first live portal with a device."""
        portal1 = mock.Mock(**{'get_device.return_value': None})
        portal2 = mock.Mock(**{'get_device.return_value': '/dev/nvme0n1'})
        portal3 = mock.Mock(**{'get_device.return_value': None})
        mock_live.return_value = [portal1, portal2]
        res = self.target.get_devices(only_live=True, get_one=True)
        self.assertListEqual(['/dev/nvme0n1'], res)
        mock_live.assert_called_once_with()
        mock_present.assert_not_called()
        portal1.get_device.assert_called_once_with()
        portal2.get_device.assert_called_once_with()
        portal3.get_device.assert_not_called()

    @mock.patch.object(nvmeof.Target, 'present_portals', new_callable=mock.PropertyMock)
    @mock.patch.object(nvmeof.Target, 'live_portals', new_callable=mock.PropertyMock)
    def test_get_devices_get_present(self, mock_live, mock_present):
        """Return all devices that are found."""
        portal1 = mock.Mock(**{'get_device.return_value': '/dev/nvme0n1'})
        portal2 = mock.Mock(**{'get_device.return_value': None})
        portal3 = mock.Mock(**{'get_device.return_value': '/dev/nvme1n1'})
        mock_present.return_value = [portal1, portal2, portal3]
        res = self.target.get_devices(only_live=False)
        self.assertIsInstance(res, list)
        self.assertEqual({'/dev/nvme0n1', '/dev/nvme1n1'}, set(res))
        mock_present.assert_called_once_with()
        mock_live.assert_not_called()
        portal1.get_device.assert_called_once_with()
        portal2.get_device.assert_called_once_with()
        portal3.get_device.assert_called_once_with()

    @mock.patch.object(nvmeof.Target, 'get_devices')
    def test_find_device_not_found(self, mock_get_devs):
        """Finding a devices tries up to 5 times before giving up."""
        mock_get_devs.return_value = []
        self.assertRaises(exception.VolumeDeviceNotFound, self.target.find_device)
        self.assertEqual(5, mock_get_devs.call_count)
        mock_get_devs.assert_has_calls(5 * [mock.call(only_live=True, get_one=True)])

    @mock.patch.object(nvmeof.Target, 'get_devices')
    def test_find_device_first_found(self, mock_get_devs):
        """Returns the first device found."""
        mock_get_devs.return_value = ['/dev/nvme0n1']
        res = self.target.find_device()
        mock_get_devs.assert_called_once_with(only_live=True, get_one=True)
        self.assertEqual('/dev/nvme0n1', res)

    @mock.patch.object(nvmeof.Target, '_get_nvme_devices')
    def test_get_device_path_by_initial_devices(self, mock_get_devs):
        """There's a new device since we started, return it."""
        self.target.portals[0].controller = 'nvme0'
        self.target.portals[1].controller = 'nvme1'
        mock_get_devs.return_value = ['/dev/nvme0n1', '/dev/nvme0n2', '/dev/nvme1n2', '/dev/nvme2n1']
        self.target.devices_on_start = ['/dev/nvme0n1', '/dev/nvme1n2']
        res = self.target.get_device_path_by_initial_devices()
        mock_get_devs.assert_called_once_with()
        self.assertEqual('/dev/nvme0n2', res)

    @mock.patch.object(nvmeof.Target, '_get_nvme_devices')
    def test_get_device_path_by_initial_devices_not_found(self, mock_get_devs):
        """There are now new devices since we started, return None."""
        self.target.portals[0].controller = 'nvme0'
        self.target.portals[1].controller = 'nvme1'
        mock_get_devs.return_value = ['/dev/nvme0n1', '/dev/nvme1n2']
        self.target.devices_on_start = ['/dev/nvme0n1', '/dev/nvme1n2']
        res = self.target.get_device_path_by_initial_devices()
        mock_get_devs.assert_called_once_with()
        self.assertIsNone(res)

    @mock.patch.object(nvmeof, 'blk_property')
    @mock.patch.object(nvmeof.Target, '_get_nvme_devices')
    def test_get_device_path_by_initial_devices_multiple(self, mock_get_devs, mock_property):
        """There are multiple new devices, but they are the same volume."""
        self.target.portals[0].controller = 'nvme0'
        self.target.portals[1].controller = 'nvme1'
        mock_property.return_value = 'uuid'
        mock_get_devs.return_value = ['/dev/nvme0n1', '/dev/nvme0n2', '/dev/nvme1n1', '/dev/nvme1n2']
        self.target.devices_on_start = ['/dev/nvme0n1', '/dev/nvme1n1']
        res = self.target.get_device_path_by_initial_devices()
        mock_get_devs.assert_called_once_with()
        self.assertEqual(2, mock_property.call_count)
        mock_property.assert_has_calls([mock.call('uuid', 'nvme0n2'), mock.call('uuid', 'nvme1n2')], any_order=True)
        self.assertIn(res, ['/dev/nvme0n2', '/dev/nvme1n2'])

    @mock.patch.object(nvmeof, 'blk_property')
    @mock.patch.object(nvmeof.Target, '_get_nvme_devices')
    def test_get_device_path_by_initial_devices_multiple_different(self, mock_get_devs, mock_property):
        """There are multiple new devices and they are different."""
        self.target.portals[0].controller = 'nvme0'
        self.target.portals[1].controller = 'nvme1'
        mock_property.side_effect = ('uuid1', 'uuid2')
        mock_get_devs.return_value = ['/dev/nvme0n1', '/dev/nvme0n2', '/dev/nvme1n1', '/dev/nvme1n2']
        self.target.devices_on_start = ['/dev/nvme0n1', '/dev/nvme1n1']
        res = self.target.get_device_path_by_initial_devices()
        mock_get_devs.assert_called_once_with()
        self.assertEqual(2, mock_property.call_count)
        mock_property.assert_has_calls([mock.call('uuid', 'nvme0n2'), mock.call('uuid', 'nvme1n2')], any_order=True)
        self.assertIsNone(res)