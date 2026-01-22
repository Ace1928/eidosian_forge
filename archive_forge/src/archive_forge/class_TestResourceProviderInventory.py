from openstack.placement.v1 import resource_provider_inventory
from openstack.tests.unit import base
class TestResourceProviderInventory(base.TestCase):

    def test_basic(self):
        sot = resource_provider_inventory.ResourceProviderInventory()
        self.assertIsNone(sot.resource_key)
        self.assertIsNone(sot.resources_key)
        self.assertEqual('/resource_providers/%(resource_provider_id)s/inventories', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertFalse(sot.allow_patch)
        self.assertDictEqual({}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = resource_provider_inventory.ResourceProviderInventory(**FAKE)
        self.assertEqual(FAKE['allocation_ratio'], sot.allocation_ratio)
        self.assertEqual(FAKE['max_unit'], sot.max_unit)
        self.assertEqual(FAKE['min_unit'], sot.min_unit)
        self.assertEqual(FAKE['reserved'], sot.reserved)
        self.assertEqual(FAKE['step_size'], sot.step_size)
        self.assertEqual(FAKE['total'], sot.total)