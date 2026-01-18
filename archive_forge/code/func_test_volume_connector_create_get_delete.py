from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_volume_connector_create_get_delete(self):
    self.conn.baremetal.set_node_provision_state(self.node, 'manage', wait=True)
    self.conn.baremetal.set_node_power_state(self.node, 'power off')
    volume_connector = self.create_volume_connector(connector_id='iqn.2017-07.org.openstack:01:d9a51732c3f', type='iqn')
    loaded = self.conn.baremetal.get_volume_connector(volume_connector.id)
    self.assertEqual(loaded.id, volume_connector.id)
    self.assertIsNotNone(loaded.node_id)
    with_fields = self.conn.baremetal.get_volume_connector(volume_connector.id, fields=['uuid', 'extra'])
    self.assertEqual(volume_connector.id, with_fields.id)
    self.assertIsNone(with_fields.node_id)
    self.conn.baremetal.delete_volume_connector(volume_connector, ignore_missing=False)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_volume_connector, volume_connector.id)