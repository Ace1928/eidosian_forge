import collections
import copy
import uuid
from osc_placement.tests.functional import base
def _setup_two_resource_providers_in_aggregate(self):
    rps = []
    invs = []
    inventory2 = ['VCPU=8', 'VCPU:max_unit=4', 'VCPU:allocation_ratio=16.0', 'MEMORY_MB=1024', 'MEMORY_MB:reserved=256', 'MEMORY_MB:allocation_ratio=2.5', 'DISK_GB=16', 'DISK_GB:allocation_ratio=1.5', 'DISK_GB:min_unit=2', 'DISK_GB:step_size=2']
    inventory1 = inventory2 + ['VGPU=8', 'VGPU:allocation_ratio=1.0', 'VGPU:min_unit=2', 'VGPU:step_size=2']
    for i, inventory in enumerate([inventory1, inventory2]):
        rps.append(self.resource_provider_create())
        resp = self.resource_inventory_set(rps[i]['uuid'], *inventory)
        self.assertNotIn('resource_provider', resp)
        invs.append({r['resource_class']: r for r in resp})
    agg = str(uuid.uuid4())
    for rp in rps:
        self.resource_provider_aggregate_set(rp['uuid'], agg)
    return (rps, agg, invs)