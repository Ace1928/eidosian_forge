import uuid
from osc_placement.tests.functional import base
class TestAllocation138(TestAllocation128):
    """Tests allocation set command with --os-placement-api-version 1.38.

    The 1.38 microversion adds the consumer_type parameter to the
    GET and PUT /allocations/{consumer_id} APIs
    """
    VERSION = '1.38'

    def test_allocation_update(self):
        consumer_uuid = str(uuid.uuid4())
        project_uuid = str(uuid.uuid4())
        user_uuid = str(uuid.uuid4())
        created_alloc = self.resource_allocation_set(consumer_uuid, ['rp={},VCPU=2'.format(self.rp1['uuid']), 'rp={},MEMORY_MB=512'.format(self.rp1['uuid'])], project_id=project_uuid, user_id=user_uuid, consumer_type='INSTANCE')
        retrieved_alloc = self.resource_allocation_show(consumer_uuid)
        expected = [{'resource_provider': self.rp1['uuid'], 'generation': 2, 'project_id': project_uuid, 'user_id': user_uuid, 'resources': {'VCPU': 2, 'MEMORY_MB': 512}, 'consumer_type': 'INSTANCE'}]
        self.assertEqual(expected, created_alloc)
        self.assertEqual(expected, retrieved_alloc)
        updated_alloc = self.resource_allocation_set(consumer_uuid, ['rp={},VCPU=4'.format(self.rp1['uuid']), 'rp={},MEMORY_MB=1024'.format(self.rp1['uuid'])], project_id=project_uuid, user_id=user_uuid, consumer_type='MIGRATION')
        expected[0].update({'generation': expected[0]['generation'] + 1, 'resources': {'VCPU': 4, 'MEMORY_MB': 1024}, 'consumer_type': 'MIGRATION'})
        self.assertEqual(expected, updated_alloc)

    def test_allocation_update_to_empty(self):
        consumer_uuid = str(uuid.uuid4())
        project_uuid = str(uuid.uuid4())
        user_uuid = str(uuid.uuid4())
        self.resource_allocation_set(consumer_uuid, ['rp={},VCPU=2'.format(self.rp1['uuid'])], project_id=project_uuid, user_id=user_uuid, consumer_type='INSTANCE')
        result = self.resource_allocation_unset(consumer_uuid, columns=('resources', 'consumer_type'))
        self.assertEqual([], result)