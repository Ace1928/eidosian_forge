from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_volume_target_list(self):
    node2 = self.create_node(name='test-node')
    self.conn.baremetal.set_node_provision_state(node2, 'manage', wait=True)
    self.conn.baremetal.set_node_power_state(node2, 'power off')
    self.conn.baremetal.set_node_provision_state(self.node, 'manage', wait=True)
    self.conn.baremetal.set_node_power_state(self.node, 'power off')
    vt1 = self.create_volume_target(boot_index=0, volume_id='bd4d008c-7d31-463d-abf9-6c23d9d55f7f', node_id=node2.id, volume_type='iscsi')
    vt2 = self.create_volume_target(boot_index=0, volume_id='04452bed-5367-4202-8bf5-de4335ac57c2', node_id=self.node.id, volume_type='iscsi')
    vts = self.conn.baremetal.volume_targets(node=self.node.id)
    self.assertEqual([v.id for v in vts], [vt2.id])
    vts = self.conn.baremetal.volume_targets(node=node2.id)
    self.assertEqual([v.id for v in vts], [vt1.id])
    vts = self.conn.baremetal.volume_targets(node='test-node')
    self.assertEqual([v.id for v in vts], [vt1.id])
    vts_with_details = self.conn.baremetal.volume_targets(details=True)
    for i in vts_with_details:
        self.assertIsNotNone(i.id)
        self.assertIsNotNone(i.volume_type)
    vts_with_fields = self.conn.baremetal.volume_targets(fields=['uuid', 'node_uuid'])
    for i in vts_with_fields:
        self.assertIsNotNone(i.id)
        self.assertIsNone(i.volume_type)
        self.assertIsNotNone(i.node_id)