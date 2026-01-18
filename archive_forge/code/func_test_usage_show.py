import operator
import uuid
from osc_placement.tests.functional import base
def test_usage_show(self):
    consumer_uuid = str(uuid.uuid4())
    rp = self.resource_provider_create()
    self.resource_inventory_set(rp['uuid'], 'VCPU=4', 'VCPU:max_unit=4', 'MEMORY_MB=1024', 'MEMORY_MB:max_unit=1024')
    self.assertEqual([{'resource_class': 'MEMORY_MB', 'usage': 0}, {'resource_class': 'VCPU', 'usage': 0}], sorted(self.resource_provider_show_usage(rp['uuid']), key=operator.itemgetter('resource_class')))
    self.resource_allocation_set(consumer_uuid, ['rp={},VCPU=2'.format(rp['uuid']), 'rp={},MEMORY_MB=512'.format(rp['uuid'])])
    self.assertEqual([{'resource_class': 'MEMORY_MB', 'usage': 512}, {'resource_class': 'VCPU', 'usage': 2}], sorted(self.resource_provider_show_usage(rp['uuid']), key=operator.itemgetter('resource_class')))