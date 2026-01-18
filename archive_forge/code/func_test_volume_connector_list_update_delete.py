from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_volume_connector_list_update_delete(self):
    self.conn.baremetal.set_node_provision_state(self.node, 'manage', wait=True)
    self.conn.baremetal.set_node_power_state(self.node, 'power off')
    self.create_volume_connector(connector_id='iqn.2020-07.org.openstack:02:d9451472ce2', node_id=self.node.id, type='iqn', extra={'foo': 'bar'})
    volume_connector = next(self.conn.baremetal.volume_connectors(details=True, node=self.node.id))
    self.assertEqual(volume_connector.extra, {'foo': 'bar'})
    self.conn.baremetal.update_volume_connector(volume_connector, extra={'foo': 42})
    self.conn.baremetal.delete_volume_connector(volume_connector, ignore_missing=False)