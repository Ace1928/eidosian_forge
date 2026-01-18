import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_replace_previous_values(self):
    """Test each new set call replaces previous inventories totally."""
    rp = self.resource_provider_create()
    self.resource_inventory_set(rp['uuid'], 'DISK_GB=16')
    self.resource_inventory_set(rp['uuid'], 'MEMORY_MB=16', 'VCPU=32')
    resp = self.resource_inventory_list(rp['uuid'])
    inv = {r['resource_class']: r for r in resp}
    self.assertNotIn('DISK_GB', inv)
    self.assertIn('VCPU', inv)
    self.assertIn('MEMORY_MB', inv)