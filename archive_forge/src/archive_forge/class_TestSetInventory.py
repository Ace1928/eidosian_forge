import collections
import copy
import uuid
from osc_placement.tests.functional import base
class TestSetInventory(base.BaseTestCase):

    def test_fail_if_no_rp(self):
        exc = self.assertRaises(base.CommandException, self.openstack, 'resource provider inventory set')
        self.assertIn(base.ARGUMENTS_MISSING, str(exc))

    def test_set_empty_inventories(self):
        rp = self.resource_provider_create()
        self.assertEqual([], self.resource_inventory_set(rp['uuid']))

    def test_fail_if_incorrect_resource(self):
        rp = self.resource_provider_create()
        exc = self.assertRaises(base.CommandException, self.resource_inventory_set, rp['uuid'], 'VCPU')
        self.assertIn('must have "name=value"', str(exc))
        exc = self.assertRaises(base.CommandException, self.resource_inventory_set, rp['uuid'], 'VCPU==')
        self.assertIn('must have "name=value"', str(exc))
        exc = self.assertRaises(base.CommandException, self.resource_inventory_set, rp['uuid'], '=10')
        self.assertIn('must be not empty', str(exc))
        exc = self.assertRaises(base.CommandException, self.resource_inventory_set, rp['uuid'], 'v=')
        self.assertIn('must be not empty', str(exc))
        exc = self.assertRaises(base.CommandException, self.resource_inventory_set, rp['uuid'], 'UNKNOWN_CPU=16')
        self.assertIn('Unknown resource class', str(exc))
        exc = self.assertRaises(base.CommandException, self.resource_inventory_set, rp['uuid'], 'VCPU:fake=16')
        self.assertIn('Unknown inventory field', str(exc))

    def test_set_multiple_classes(self):
        rp = self.resource_provider_create()
        resp = self.resource_inventory_set(rp['uuid'], 'VCPU=8', 'VCPU:max_unit=4', 'MEMORY_MB=1024', 'MEMORY_MB:reserved=256', 'DISK_GB=16', 'DISK_GB:allocation_ratio=1.5', 'DISK_GB:min_unit=2', 'DISK_GB:step_size=2')

        def check(inventories):
            self.assertEqual(8, inventories['VCPU']['total'])
            self.assertEqual(4, inventories['VCPU']['max_unit'])
            self.assertEqual(1024, inventories['MEMORY_MB']['total'])
            self.assertEqual(256, inventories['MEMORY_MB']['reserved'])
            self.assertEqual(16, inventories['DISK_GB']['total'])
            self.assertEqual(2, inventories['DISK_GB']['min_unit'])
            self.assertEqual(2, inventories['DISK_GB']['step_size'])
            self.assertEqual(1.5, inventories['DISK_GB']['allocation_ratio'])
        check({r['resource_class']: r for r in resp})
        resp = self.resource_inventory_list(rp['uuid'])
        check({r['resource_class']: r for r in resp})

    def test_set_known_and_unknown_class(self):
        rp = self.resource_provider_create()
        exc = self.assertRaises(base.CommandException, self.resource_inventory_set, rp['uuid'], 'VCPU=8', 'UNKNOWN=4')
        self.assertIn('Unknown resource class', str(exc))
        self.assertEqual([], self.resource_inventory_list(rp['uuid']))

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

    def test_delete_via_set(self):
        rp = self.resource_provider_create()
        self.resource_inventory_set(rp['uuid'], 'DISK_GB=16')
        self.resource_inventory_set(rp['uuid'])
        self.assertEqual([], self.resource_inventory_list(rp['uuid']))

    def test_fail_if_incorrect_parameters_set_class_inventory(self):
        exc = self.assertRaises(base.CommandException, self.openstack, 'resource provider inventory class set')
        self.assertIn(base.ARGUMENTS_MISSING, str(exc))
        exc = self.assertRaises(base.CommandException, self.openstack, 'resource provider inventory class set fake_uuid')
        self.assertIn(base.ARGUMENTS_MISSING, str(exc))
        exc = self.assertRaises(base.CommandException, self.openstack, 'resource provider inventory class set fake_uuid fake_class --total 5 --unknown 1')
        self.assertIn('unrecognized arguments', str(exc))
        rp = self.resource_provider_create()
        exc = self.assertRaises(base.CommandException, self.openstack, 'resource provider inventory class set %s VCPU' % rp['uuid'])
        self.assertIn(base.ARGUMENTS_REQUIRED % '--total', str(exc))

    def test_set_inventory_for_resource_class(self):
        rp = self.resource_provider_create()
        self.resource_inventory_set(rp['uuid'], 'MEMORY_MB=16', 'VCPU=32')
        self.resource_inventory_class_set(rp['uuid'], 'MEMORY_MB', total=128, step_size=16)
        resp = self.resource_inventory_list(rp['uuid'])
        inv = {r['resource_class']: r for r in resp}
        self.assertEqual(128, inv['MEMORY_MB']['total'])
        self.assertEqual(16, inv['MEMORY_MB']['step_size'])
        self.assertEqual(32, inv['VCPU']['total'])

    def test_fail_aggregate_arg_version_handling(self):
        agg = str(uuid.uuid4())
        self.assertCommandFailed('Operation or argument is not supported with version 1.0; requires at least version 1.3', self.resource_inventory_set, agg, 'MEMORY_MB=16', aggregate=True)

    def test_amend_multiple_classes(self):
        rp = self.resource_provider_create()
        resp = self.resource_inventory_set(rp['uuid'], 'VCPU=8', 'VCPU:max_unit=4', 'MEMORY_MB=1024', 'MEMORY_MB:reserved=256', 'DISK_GB=16', 'DISK_GB:allocation_ratio=1.5', 'DISK_GB:min_unit=2', 'DISK_GB:step_size=2', amend=True)

        def check(inventories):
            self.assertEqual(8, inventories['VCPU']['total'])
            self.assertEqual(4, inventories['VCPU']['max_unit'])
            self.assertEqual(1024, inventories['MEMORY_MB']['total'])
            self.assertEqual(256, inventories['MEMORY_MB']['reserved'])
            self.assertEqual(16, inventories['DISK_GB']['total'])
            self.assertEqual(1.5, inventories['DISK_GB']['allocation_ratio'])
            self.assertEqual(2, inventories['DISK_GB']['min_unit'])
            self.assertEqual(2, inventories['DISK_GB']['step_size'])
        inventories = {r['resource_class']: r for r in resp}
        check(inventories)
        resp = self.resource_inventory_list(rp['uuid'])
        inventories = {r['resource_class']: r for r in resp}
        check(inventories)
        resp = self.resource_inventory_set(rp['uuid'], 'VCPU:allocation_ratio=5.0', amend=True)
        inventories = {r['resource_class']: r for r in resp}
        check(inventories)
        self.assertEqual(5.0, inventories['VCPU']['allocation_ratio'])
        resp = self.resource_inventory_list(rp['uuid'])
        inventories = {r['resource_class']: r for r in resp}
        check(inventories)
        self.assertEqual(5.0, inventories['VCPU']['allocation_ratio'])

    def test_dry_run(self):
        rp = self.resource_provider_create()
        resp = self.resource_inventory_set(rp['uuid'], 'VCPU=8', 'VCPU:max_unit=4', 'MEMORY_MB=1024', 'MEMORY_MB:reserved=256', 'DISK_GB=16', 'DISK_GB:allocation_ratio=1.5', 'DISK_GB:min_unit=2', 'DISK_GB:step_size=2', dry_run=True)

        def check(inventories):
            self.assertEqual(8, inventories['VCPU']['total'])
            self.assertEqual(4, inventories['VCPU']['max_unit'])
            self.assertEqual(1024, inventories['MEMORY_MB']['total'])
            self.assertEqual(256, inventories['MEMORY_MB']['reserved'])
            self.assertEqual(16, inventories['DISK_GB']['total'])
            self.assertEqual(2, inventories['DISK_GB']['min_unit'])
            self.assertEqual(2, inventories['DISK_GB']['step_size'])
            self.assertEqual(1.5, inventories['DISK_GB']['allocation_ratio'])
        check({r['resource_class']: r for r in resp})
        resp = self.resource_inventory_list(rp['uuid'])
        self.assertEqual([], resp)