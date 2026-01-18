from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_volume_target_patch(self):
    vol_targ_id = '04452bed-5367-4202-9cg6-de4335ac53h7'
    self.conn.baremetal.set_node_provision_state(self.node, 'manage', wait=True)
    self.conn.baremetal.set_node_power_state(self.node, 'power off')
    volume_target = self.create_volume_target(boot_index=0, volume_id=vol_targ_id, node_id=self.node.id, volume_type='isci')
    volume_target = self.conn.baremetal.patch_volume_target(volume_target, dict(path='/extra/answer', op='add', value=42))
    self.assertEqual({'answer': 42}, volume_target.extra)
    self.assertEqual(vol_targ_id, volume_target.volume_id)
    volume_target = self.conn.baremetal.get_volume_target(volume_target.id)
    self.assertEqual({'answer': 42}, volume_target.extra)