from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_volume_target_update(self):
    self.conn.baremetal.set_node_provision_state(self.node, 'manage', wait=True)
    self.conn.baremetal.set_node_power_state(self.node, 'power off')
    volume_target = self.create_volume_target(boot_index=0, volume_id='04452bed-5367-4202-8bf5-de4335ac53h7', node_id=self.node.id, volume_type='isci')
    volume_target.extra = {'answer': 42}
    volume_target = self.conn.baremetal.update_volume_target(volume_target)
    self.assertEqual({'answer': 42}, volume_target.extra)
    volume_target = self.conn.baremetal.get_volume_target(volume_target.id)
    self.assertEqual({'answer': 42}, volume_target.extra)