from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_volume_target_fields(self):
    self.create_node()
    self.conn.baremetal.set_node_provision_state(self.node, 'manage', wait=True)
    self.conn.baremetal.set_node_power_state(self.node, 'power off')
    self.create_volume_target(boot_index=0, volume_id='04452bed-5367-4202-8bf5-99ae634d8971', node_id=self.node.id, volume_type='iscsi')
    result = self.conn.baremetal.volume_targets(fields=['uuid', 'node_id'])
    for item in result:
        self.assertIsNotNone(item.id)