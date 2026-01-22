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
class NVMeOFConnPropsTestCase(test_base.TestCase):

    @mock.patch.object(nvmeof.Target, 'factory')
    def test_init_old_props(self, mock_target):
        """Test init with old format connection properties."""
        conn_props = {'nqn': 'nqn_value', 'transport_type': 'rdma', 'target_portal': 'portal_value', 'target_port': 'port_value', 'volume_nguid': 'nguid', 'ns_id': 'nsid', 'host_nqn': 'host_nqn_value', 'qos_specs': None, 'access_mode': 'rw', 'encrypted': False, 'cacheable': True, 'discard': True}
        res = nvmeof.NVMeOFConnProps(conn_props, mock.sentinel.find_controllers)
        self.assertFalse(res.is_replicated)
        self.assertIsNone(res.qos_specs)
        self.assertFalse(res.readonly)
        self.assertFalse(res.encrypted)
        self.assertTrue(res.cacheable)
        self.assertTrue(res.discard)
        self.assertIsNone(res.alias)
        self.assertIsNone(res.cinder_volume_id)
        mock_target.assert_called_once_with(source_conn_props=res, find_controllers=mock.sentinel.find_controllers, volume_nguid='nguid', ns_id='nsid', host_nqn='host_nqn_value', portals=[('portal_value', 'port_value', 'rdma')], vol_uuid=None, target_nqn='nqn_value', qos_specs=None, access_mode='rw', encrypted=False, cacheable=True, discard=True)
        self.assertListEqual([mock_target.return_value], res.targets)

    @ddt.data('vol_uuid', 'ns_id', 'volume_nguid')
    @mock.patch.object(nvmeof.Target, 'factory')
    def test_init_new_props_unreplicated(self, id_name, mock_target):
        """Test init with new format connection properties but no replicas."""
        conn_props = {'target_nqn': 'nqn_value', id_name: 'uuid', 'portals': [('portal1', 'port_value', 'RoCEv2'), ('portal2', 'port_value', 'anything')], 'qos_specs': None, 'access_mode': 'rw', 'encrypted': False, 'cacheable': True, 'discard': True}
        res = nvmeof.NVMeOFConnProps(conn_props, mock.sentinel.find_controllers)
        self.assertFalse(res.is_replicated)
        self.assertIsNone(res.qos_specs)
        self.assertFalse(res.readonly)
        self.assertFalse(res.encrypted)
        self.assertTrue(res.cacheable)
        self.assertTrue(res.discard)
        self.assertIsNone(res.alias)
        self.assertIsNone(res.cinder_volume_id)
        kw_id_arg = {id_name: 'uuid'}
        mock_target.assert_called_once_with(source_conn_props=res, find_controllers=mock.sentinel.find_controllers, target_nqn='nqn_value', portals=[('portal1', 'port_value', 'RoCEv2'), ('portal2', 'port_value', 'anything')], qos_specs=None, access_mode='rw', encrypted=False, cacheable=True, discard=True, **kw_id_arg)
        self.assertListEqual([mock_target.return_value], res.targets)

    @mock.patch.object(nvmeof.Target, 'factory')
    def test_init_new_props_replicated(self, mock_target):
        """Test init with new format connection properties with replicas."""
        conn_props = {'vol_uuid': VOL_UUID_NO_HYPHENS, 'alias': 'raid_alias', 'replica_count': 2, 'volume_replicas': [{'target_nqn': 'nqn1', 'vol_uuid': VOL_UUID1, 'portals': [['portal1', 'port_value', 'RoCEv2'], ['portal2', 'port_value', 'anything']]}, {'target_nqn': 'nqn2', 'vol_uuid': VOL_UUID2, 'portals': [['portal4', 'port_value', 'anything'], ['portal3', 'port_value', 'RoCEv2']]}], 'qos_specs': None, 'access_mode': 'ro', 'encrypted': True, 'cacheable': False, 'discard': False}
        targets = [mock.Mock(), mock.Mock()]
        mock_target.side_effect = targets
        res = nvmeof.NVMeOFConnProps(conn_props, mock.sentinel.find_controllers)
        self.assertTrue(res.is_replicated)
        self.assertIsNone(res.qos_specs)
        self.assertTrue(res.readonly)
        self.assertTrue(res.encrypted)
        self.assertFalse(res.cacheable)
        self.assertFalse(res.discard)
        self.assertEqual('raid_alias', res.alias)
        self.assertEqual(VOL_UUID, res.cinder_volume_id)
        self.assertEqual(2, mock_target.call_count)
        call_1 = dict(source_conn_props=res, find_controllers=mock.sentinel.find_controllers, vol_uuid=VOL_UUID1, target_nqn='nqn1', portals=[['portal1', 'port_value', 'RoCEv2'], ['portal2', 'port_value', 'anything']])
        call_2 = dict(source_conn_props=res, find_controllers=mock.sentinel.find_controllers, vol_uuid=VOL_UUID2, target_nqn='nqn2', portals=[['portal4', 'port_value', 'anything'], ['portal3', 'port_value', 'RoCEv2']])
        mock_target.assert_has_calls([mock.call(**call_1), mock.call(**call_2)])
        self.assertListEqual(targets, res.targets)

    @mock.patch.object(nvmeof.Target, 'factory')
    def test_get_devices(self, mock_target):
        """Connector get devices gets devices from all its portals."""
        conn_props = {'vol_uuid': VOL_UUID, 'alias': 'raid_alias', 'replica_count': 2, 'volume_replicas': [{'target_nqn': 'nqn1', 'vol_uuid': VOL_UUID1, 'portals': [['portal1', 'port_value', 'RoCEv2'], ['portal2', 'port_value', 'anything']]}, {'target_nqn': VOL_UUID2, 'vol_uuid': 'uuid2', 'portals': [['portal4', 'port_value', 'anything'], ['portal3', 'port_value', 'RoCEv2']]}]}
        targets = [mock.Mock(), mock.Mock()]
        targets[0].get_devices.return_value = []
        targets[1].get_devices.return_value = ['/dev/nvme0n1', '/dev/nvme0n2']
        mock_target.side_effect = targets
        conn_props_instance = nvmeof.NVMeOFConnProps(conn_props)
        res = conn_props_instance.get_devices(mock.sentinel.only_live)
        self.assertListEqual(['/dev/nvme0n1', '/dev/nvme0n2'], res)

    @mock.patch.object(nvmeof.Target, 'factory')
    def test_from_dictionary_parameter(self, mock_target):
        """Decorator converts dict into connection properties instance."""

        class Connector(object):

            @nvmeof.NVMeOFConnProps.from_dictionary_parameter
            def connect_volume(my_self, connection_properties):
                self.assertIsInstance(connection_properties, nvmeof.NVMeOFConnProps)
                return 'result'
        conn = Connector()
        conn_props = {'target_nqn': 'nqn_value', 'vol_uuid': 'uuid', 'portals': [('portal1', 'port_value', 'RoCEv2'), ('portal2', 'port_value', 'anything')]}
        res = conn.connect_volume(conn_props)
        self.assertEqual('result', res)