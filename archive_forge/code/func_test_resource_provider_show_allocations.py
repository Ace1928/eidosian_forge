import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_show_allocations(self):
    consumer_uuid = str(uuid.uuid4())
    allocs = {consumer_uuid: {'resources': {'VCPU': 2}}}
    created = self.resource_provider_create()
    self.resource_inventory_set(created['uuid'], 'VCPU=4', 'VCPU:max_unit=4')
    self.resource_allocation_set(consumer_uuid, ['rp={},VCPU=2'.format(created['uuid'])])
    expected = dict(created, allocations=allocs, generation=2)
    retrieved = self.resource_provider_show(created['uuid'], allocations=True)
    self.assertEqual(expected, retrieved)