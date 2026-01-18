from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_volume_connector_fields(self):
    self.create_node()
    self.conn.baremetal.set_node_provision_state(self.node, 'manage', wait=True)
    self.conn.baremetal.set_node_power_state(self.node, 'power off')
    self.create_volume_connector(connector_id='iqn.2018-08.org.openstack:04:de45f37c48', node_id=self.node.id, type='iqn')
    result = self.conn.baremetal.volume_connectors(fields=['uuid', 'node_id'])
    for item in result:
        self.assertIsNotNone(item.id)
        self.assertIsNone(item.connector_id)