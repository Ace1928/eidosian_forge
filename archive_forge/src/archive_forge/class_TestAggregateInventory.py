import collections
import copy
import uuid
from osc_placement.tests.functional import base
class TestAggregateInventory(base.BaseTestCase):
    VERSION = '1.3'

    def _test_dry_run(self, agg, rps, old_inventories, amend=False):
        new_resources = ['VCPU:allocation_ratio=5.0', 'MEMORY_MB:allocation_ratio=6.0', 'DISK_GB:allocation_ratio=7.0']
        resp = self.resource_inventory_set(agg, *new_resources, aggregate=True, amend=amend, dry_run=True)
        inventories = old_inventories if amend else [{}] * len(old_inventories)
        new_inventories = self._get_expected_inventories(inventories, new_resources)
        expected = {}
        for rp, inventory in zip(rps, new_inventories):
            for rc, inv in inventory.items():
                inv['resource_provider'] = rp['uuid']
                for key in ('max_unit', 'min_unit', 'reserved', 'step_size', 'total', 'reserved', 'step_size'):
                    if key not in inv:
                        inv[key] = ''
            expected[rp['uuid']] = inventory
        resp_dict = collections.defaultdict(dict)
        for row in resp:
            resp_dict[row['resource_provider']][row['resource_class']] = row
        self.assertEqual(expected, resp_dict)
        for i, rp in enumerate(rps):
            resp = self.resource_inventory_list(rp['uuid'])
            self.assertDictEqual(old_inventories[i], {r['resource_class']: r for r in resp})

    def _get_expected_inventories(self, old_inventories, resources):
        new_inventories = []
        for old_inventory in old_inventories:
            new_inventory = collections.defaultdict(dict)
            new_inventory.update(copy.deepcopy(old_inventory))
            for resource in resources:
                rc, keyval = resource.split(':')
                key, val = keyval.split('=')
                val = float(val) if '.' in val else int(val)
                new_inventory[rc][key] = val
                if 'resource_class' not in new_inventory[rc]:
                    new_inventory[rc]['resource_class'] = rc
            new_inventories.append(new_inventory)
        return new_inventories

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

    def test_fail_if_no_rps_in_aggregate(self):
        nonexistent_agg = str(uuid.uuid4())
        exc = self.assertRaises(base.CommandException, self.resource_inventory_set, nonexistent_agg, 'VCPU=8', aggregate=True)
        self.assertIn('No resource providers found in aggregate with uuid {}'.format(nonexistent_agg), str(exc))

    def test_with_aggregate_one_fails(self):
        rps, agg, _invs = self._setup_two_resource_providers_in_aggregate()
        self.resource_class_create('CUSTOM_FOO')
        rp1_uuid = rps[0]['uuid']
        rp1_inv = self.resource_inventory_set(rp1_uuid, 'CUSTOM_FOO=1')
        consumer = str(uuid.uuid4())
        alloc = 'rp=%s,CUSTOM_FOO=1' % rp1_uuid
        self.resource_allocation_set(consumer, [alloc])
        new_resources = ['VCPU:allocation_ratio=5.0', 'VCPU:total=8']
        exc = self.assertRaises(base.CommandException, self.resource_inventory_set, agg, *new_resources, aggregate=True)
        self.assertIn('Failed to set inventory for 1 of 2 resource providers.', str(exc))
        output = self.output.getvalue() + self.error.getvalue()
        self.assertIn('Failed to set inventory for resource provider %s:' % rp1_uuid, output)
        err_txt = "Inventory for 'CUSTOM_FOO' on resource provider '%s' in use." % rp1_uuid
        self.assertIn(err_txt, output)
        placement_defaults = ['VCPU:max_unit=2147483647', 'VCPU:min_unit=1', 'VCPU:reserved=0', 'VCPU:step_size=1']
        new_inventories = self._get_expected_inventories([{}], new_resources + placement_defaults)
        resp = self.resource_inventory_list(rps[1]['uuid'])
        self.assertDictEqual(new_inventories[0], {r['resource_class']: r for r in resp})
        resp = self.resource_inventory_list(rp1_uuid)
        self.assertDictEqual({r['resource_class']: r for r in rp1_inv}, {r['resource_class']: r for r in resp})

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

    def test_with_aggregate(self):
        self._test_with_aggregate()

    def test_amend_with_aggregate(self):
        self._test_with_aggregate(amend=True)