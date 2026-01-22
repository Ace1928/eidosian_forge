import uuid
from osc_placement.tests.functional import base
class TestAllocation18(base.BaseTestCase):
    VERSION = '1.8'

    def test_allocation_create(self):
        consumer_uuid = str(uuid.uuid4())
        project_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        rp1 = self.resource_provider_create()
        self.resource_inventory_set(rp1['uuid'], 'VCPU=4', 'VCPU:max_unit=4', 'MEMORY_MB=1024', 'MEMORY_MB:max_unit=1024')
        created_alloc = self.resource_allocation_set(consumer_uuid, ['rp={},VCPU=2'.format(rp1['uuid']), 'rp={},MEMORY_MB=512'.format(rp1['uuid'])], project_id=project_id, user_id=user_id)
        retrieved_alloc = self.resource_allocation_show(consumer_uuid)
        expected = [{'resource_provider': rp1['uuid'], 'generation': 2, 'resources': {'VCPU': 2, 'MEMORY_MB': 512}}]
        self.assertEqual(expected, created_alloc)
        self.assertEqual(expected, retrieved_alloc)