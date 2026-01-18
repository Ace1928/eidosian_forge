import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_set_inventory_for_resource_class(self):
    rp = self.resource_provider_create()
    self.resource_inventory_set(rp['uuid'], 'MEMORY_MB=16', 'VCPU=32')
    self.resource_inventory_class_set(rp['uuid'], 'MEMORY_MB', total=128, step_size=16)
    resp = self.resource_inventory_list(rp['uuid'])
    inv = {r['resource_class']: r for r in resp}
    self.assertEqual(128, inv['MEMORY_MB']['total'])
    self.assertEqual(16, inv['MEMORY_MB']['step_size'])
    self.assertEqual(32, inv['VCPU']['total'])