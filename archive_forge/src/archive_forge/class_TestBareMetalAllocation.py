import random
from openstack import exceptions
from openstack.tests.functional.baremetal import base
class TestBareMetalAllocation(Base):
    min_microversion = '1.52'

    def test_allocation_create_get_delete(self):
        allocation = self.create_allocation(resource_class=self.resource_class)
        self.assertEqual('allocating', allocation.state)
        self.assertIsNone(allocation.node_id)
        self.assertIsNone(allocation.last_error)
        loaded = self.conn.baremetal.wait_for_allocation(allocation)
        self.assertEqual(loaded.id, allocation.id)
        self.assertEqual('active', allocation.state)
        self.assertEqual(self.node.id, allocation.node_id)
        self.assertIsNone(allocation.last_error)
        with_fields = self.conn.baremetal.get_allocation(allocation.id, fields=['uuid', 'node_uuid'])
        self.assertEqual(allocation.id, with_fields.id)
        self.assertIsNone(with_fields.state)
        node = self.conn.baremetal.get_node(self.node.id)
        self.assertEqual(allocation.id, node.allocation_id)
        self.conn.baremetal.delete_allocation(allocation, ignore_missing=False)
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_allocation, allocation.id)

    def test_allocation_list(self):
        allocation1 = self.create_allocation(resource_class=self.resource_class)
        allocation2 = self.create_allocation(resource_class=self.resource_class + '-fail')
        self.conn.baremetal.wait_for_allocation(allocation1)
        self.conn.baremetal.wait_for_allocation(allocation2, ignore_error=True)
        allocations = self.conn.baremetal.allocations()
        self.assertEqual({p.id for p in allocations}, {allocation1.id, allocation2.id})
        allocations = self.conn.baremetal.allocations(state='active')
        self.assertEqual([p.id for p in allocations], [allocation1.id])
        allocations = self.conn.baremetal.allocations(node=self.node.id)
        self.assertEqual([p.id for p in allocations], [allocation1.id])
        allocations = self.conn.baremetal.allocations(resource_class=self.resource_class + '-fail')
        self.assertEqual([p.id for p in allocations], [allocation2.id])

    def test_allocation_negative_failure(self):
        allocation = self.create_allocation(resource_class=self.resource_class + '-fail')
        self.assertRaises(exceptions.SDKException, self.conn.baremetal.wait_for_allocation, allocation)
        allocation = self.conn.baremetal.get_allocation(allocation.id)
        self.assertEqual('error', allocation.state)
        self.assertIn(self.resource_class + '-fail', allocation.last_error)

    def test_allocation_negative_non_existing(self):
        uuid = '5c9dcd04-2073-49bc-9618-99ae634d8971'
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_allocation, uuid)
        self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.delete_allocation, uuid, ignore_missing=False)
        self.assertIsNone(self.conn.baremetal.delete_allocation(uuid))

    def test_allocation_fields(self):
        self.create_allocation(resource_class=self.resource_class)
        result = self.conn.baremetal.allocations(fields=['uuid'])
        for item in result:
            self.assertIsNotNone(item.id)
            self.assertIsNone(item.resource_class)