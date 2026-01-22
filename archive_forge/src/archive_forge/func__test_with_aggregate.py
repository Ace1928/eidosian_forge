import collections
import copy
import uuid
from osc_placement.tests.functional import base
def _test_with_aggregate(self, amend=False):
    rps, agg, old_invs = self._setup_two_resource_providers_in_aggregate()
    self._test_dry_run(agg, rps, old_invs, amend=amend)
    if not amend:
        old_invs = []
        defaults = {'max_unit': 2147483647, 'min_unit': 1, 'reserved': 0, 'step_size': 1}
        default_inventory = {'VCPU': copy.deepcopy(defaults)}
        for rp in rps:
            old_invs.append(default_inventory)
    new_resources = ['VCPU:allocation_ratio=5.0', 'VCPU:total=8']
    resp = self.resource_inventory_set(agg, *new_resources, aggregate=True, amend=amend)
    for rp in resp:
        self.assertIn('resource_provider', rp)
    new_inventories = self._get_expected_inventories(old_invs, new_resources)
    for i, rp in enumerate(rps):
        resp = self.resource_inventory_list(rp['uuid'])
        self.assertDictEqual(new_inventories[i], {r['resource_class']: r for r in resp})