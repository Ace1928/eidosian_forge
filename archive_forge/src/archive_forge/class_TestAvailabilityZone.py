from openstack.network.v2 import availability_zone
from openstack.tests.unit import base
class TestAvailabilityZone(base.TestCase):

    def test_basic(self):
        sot = availability_zone.AvailabilityZone()
        self.assertEqual('availability_zone', sot.resource_key)
        self.assertEqual('availability_zones', sot.resources_key)
        self.assertEqual('/availability_zones', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertFalse(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = availability_zone.AvailabilityZone(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['resource'], sot.resource)
        self.assertEqual(EXAMPLE['state'], sot.state)