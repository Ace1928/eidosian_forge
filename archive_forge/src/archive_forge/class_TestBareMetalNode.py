import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
class TestBareMetalNode(base.BaseBaremetalTest):

    def test_node_create_get_delete(self):
        node = self.create_node(name='node-name')
        self.assertEqual(node.name, 'node-name')
        self.assertEqual(node.driver, 'fake-hardware')
        self.assertEqual(node.provision_state, 'enroll')
        self.assertFalse(node.is_maintenance)
        for call, ident in [(self.conn.baremetal.get_node, self.node_id), (self.conn.baremetal.get_node, 'node-name'), (self.conn.baremetal.find_node, self.node_id), (self.conn.baremetal.find_node, 'node-name')]:
            found = call(ident)
            self.assertEqual(node.id, found.id)
            self.assertEqual(node.name, found.name)
        with_fields = self.conn.baremetal.get_node('node-name', fields=['uuid', 'driver', 'instance_id'])
        self.assertEqual(node.id, with_fields.id)
        self.assertEqual(node.driver, with_fields.driver)
        self.assertIsNone(with_fields.name)
        self.assertIsNone(with_fields.provision_state)
        nodes = self.conn.baremetal.nodes()
        self.assertIn(node.id, [n.id for n in nodes])
        self.conn.baremetal.delete_node(node, ignore_missing=False)
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_node, self.node_id)

    def test_node_create_in_available(self):
        node = self.create_node(name='node-name', provision_state='available')
        self.assertEqual(node.name, 'node-name')
        self.assertEqual(node.driver, 'fake-hardware')
        self.assertEqual(node.provision_state, 'available')
        self.conn.baremetal.delete_node(node, ignore_missing=False)
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_node, self.node_id)

    def test_node_update(self):
        node = self.create_node(name='node-name', extra={'foo': 'bar'})
        node.name = 'new-name'
        node.extra = {'answer': 42}
        instance_uuid = str(uuid.uuid4())
        node = self.conn.baremetal.update_node(node, instance_id=instance_uuid)
        self.assertEqual('new-name', node.name)
        self.assertEqual({'answer': 42}, node.extra)
        self.assertEqual(instance_uuid, node.instance_id)
        node = self.conn.baremetal.get_node('new-name')
        self.assertEqual('new-name', node.name)
        self.assertEqual({'answer': 42}, node.extra)
        self.assertEqual(instance_uuid, node.instance_id)
        node = self.conn.baremetal.update_node(node, instance_id=None)
        self.assertIsNone(node.instance_id)
        node = self.conn.baremetal.get_node('new-name')
        self.assertIsNone(node.instance_id)

    def test_node_update_by_name(self):
        self.create_node(name='node-name', extra={'foo': 'bar'})
        instance_uuid = str(uuid.uuid4())
        node = self.conn.baremetal.update_node('node-name', instance_id=instance_uuid, extra={'answer': 42})
        self.assertEqual({'answer': 42}, node.extra)
        self.assertEqual(instance_uuid, node.instance_id)
        node = self.conn.baremetal.get_node('node-name')
        self.assertEqual({'answer': 42}, node.extra)
        self.assertEqual(instance_uuid, node.instance_id)
        node = self.conn.baremetal.update_node('node-name', instance_id=None)
        self.assertIsNone(node.instance_id)
        node = self.conn.baremetal.get_node('node-name')
        self.assertIsNone(node.instance_id)

    def test_node_patch(self):
        node = self.create_node(name='node-name', extra={'foo': 'bar'})
        node.name = 'new-name'
        instance_uuid = str(uuid.uuid4())
        node = self.conn.baremetal.patch_node(node, [dict(path='/instance_id', op='replace', value=instance_uuid), dict(path='/extra/answer', op='add', value=42)])
        self.assertEqual('new-name', node.name)
        self.assertEqual({'foo': 'bar', 'answer': 42}, node.extra)
        self.assertEqual(instance_uuid, node.instance_id)
        node = self.conn.baremetal.get_node('new-name')
        self.assertEqual('new-name', node.name)
        self.assertEqual({'foo': 'bar', 'answer': 42}, node.extra)
        self.assertEqual(instance_uuid, node.instance_id)
        node = self.conn.baremetal.patch_node(node, [dict(path='/instance_id', op='remove'), dict(path='/extra/answer', op='remove')])
        self.assertIsNone(node.instance_id)
        self.assertNotIn('answer', node.extra)
        node = self.conn.baremetal.get_node('new-name')
        self.assertIsNone(node.instance_id)
        self.assertNotIn('answer', node.extra)

    def test_node_list_update_delete(self):
        self.create_node(name='node-name', extra={'foo': 'bar'})
        node = next((n for n in self.conn.baremetal.nodes(details=True, provision_state='enroll', is_maintenance=False, associated=False) if n.name == 'node-name'))
        self.assertEqual(node.extra, {'foo': 'bar'})
        self.conn.baremetal.update_node(node, extra={'foo': 42})
        self.conn.baremetal.delete_node(node, ignore_missing=False)

    def test_node_create_in_enroll_provide(self):
        node = self.create_node()
        self.node_id = node.id
        self.assertEqual(node.driver, 'fake-hardware')
        self.assertEqual(node.provision_state, 'enroll')
        self.assertIsNone(node.power_state)
        self.assertFalse(node.is_maintenance)
        self.conn.baremetal.set_node_provision_state(node, 'manage', wait=True)
        self.assertEqual(node.provision_state, 'manageable')
        self.conn.baremetal.set_node_provision_state(node, 'provide', wait=True)
        self.assertEqual(node.provision_state, 'available')

    def test_node_create_in_enroll_provide_by_name(self):
        name = 'node-%d' % random.randint(0, 1000)
        node = self.create_node(name=name)
        self.node_id = node.id
        self.assertEqual(node.driver, 'fake-hardware')
        self.assertEqual(node.provision_state, 'enroll')
        self.assertIsNone(node.power_state)
        self.assertFalse(node.is_maintenance)
        node = self.conn.baremetal.set_node_provision_state(name, 'manage', wait=True)
        self.assertEqual(node.provision_state, 'manageable')
        node = self.conn.baremetal.set_node_provision_state(name, 'provide', wait=True)
        self.assertEqual(node.provision_state, 'available')

    def test_node_power_state(self):
        node = self.create_node()
        self.assertIsNone(node.power_state)
        self.conn.baremetal.set_node_power_state(node, 'power on', wait=True)
        node = self.conn.baremetal.get_node(node.id)
        self.assertEqual('power on', node.power_state)
        self.conn.baremetal.set_node_power_state(node, 'power off', wait=True)
        node = self.conn.baremetal.get_node(node.id)
        self.assertEqual('power off', node.power_state)

    def test_node_validate(self):
        node = self.create_node()
        result = self.conn.baremetal.validate_node(node)
        for iface in ('boot', 'deploy', 'management', 'power'):
            self.assertTrue(result[iface].result)
            self.assertFalse(result[iface].reason)

    def test_node_negative_non_existing(self):
        uuid = '5c9dcd04-2073-49bc-9618-99ae634d8971'
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_node, uuid)
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.find_node, uuid, ignore_missing=False)
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.delete_node, uuid, ignore_missing=False)
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.update_node, uuid, name='new-name')
        self.assertIsNone(self.conn.baremetal.find_node(uuid))
        self.assertIsNone(self.conn.baremetal.delete_node(uuid))

    def test_maintenance(self):
        reason = 'Prepating for taking over the world'
        node = self.create_node()
        self.assertFalse(node.is_maintenance)
        self.assertIsNone(node.maintenance_reason)
        node = self.conn.baremetal.set_node_maintenance(node)
        self.assertTrue(node.is_maintenance)
        self.assertIsNone(node.maintenance_reason)
        node = self.conn.baremetal.set_node_maintenance(node, reason)
        self.assertTrue(node.is_maintenance)
        self.assertEqual(reason, node.maintenance_reason)
        node = self.conn.baremetal.set_node_maintenance(node)
        self.assertTrue(node.is_maintenance)
        self.assertIsNone(node.maintenance_reason)
        node = self.conn.baremetal.unset_node_maintenance(node)
        self.assertFalse(node.is_maintenance)
        self.assertIsNone(node.maintenance_reason)
        node = self.conn.baremetal.set_node_maintenance(node, reason)
        self.assertTrue(node.is_maintenance)
        self.assertEqual(reason, node.maintenance_reason)

    def test_maintenance_via_update(self):
        reason = 'Prepating for taking over the world'
        node = self.create_node()
        node = self.conn.baremetal.update_node(node, is_maintenance=True)
        self.assertTrue(node.is_maintenance)
        self.assertIsNone(node.maintenance_reason)
        node = self.conn.baremetal.get_node(node.id)
        self.assertTrue(node.is_maintenance)
        self.assertIsNone(node.maintenance_reason)
        node = self.conn.baremetal.update_node(node, maintenance_reason=reason)
        self.assertTrue(node.is_maintenance)
        self.assertEqual(reason, node.maintenance_reason)
        node = self.conn.baremetal.get_node(node.id)
        self.assertTrue(node.is_maintenance)
        self.assertEqual(reason, node.maintenance_reason)
        node = self.conn.baremetal.update_node(node, is_maintenance=False)
        self.assertFalse(node.is_maintenance)
        self.assertIsNone(node.maintenance_reason)
        node = self.conn.baremetal.get_node(node.id)
        self.assertFalse(node.is_maintenance)
        self.assertIsNone(node.maintenance_reason)
        node = self.conn.baremetal.update_node(node, is_maintenance=True, maintenance_reason=reason)
        self.assertTrue(node.is_maintenance)
        self.assertEqual(reason, node.maintenance_reason)
        node = self.conn.baremetal.get_node(node.id)
        self.assertTrue(node.is_maintenance)
        self.assertEqual(reason, node.maintenance_reason)