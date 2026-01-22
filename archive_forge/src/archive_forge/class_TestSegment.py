from openstack.network.v2 import segment
from openstack.tests.unit import base
class TestSegment(base.TestCase):

    def test_basic(self):
        sot = segment.Segment()
        self.assertEqual('segment', sot.resource_key)
        self.assertEqual('segments', sot.resources_key)
        self.assertEqual('/segments', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = segment.Segment(**EXAMPLE)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['network_id'], sot.network_id)
        self.assertEqual(EXAMPLE['network_type'], sot.network_type)
        self.assertEqual(EXAMPLE['physical_network'], sot.physical_network)
        self.assertEqual(EXAMPLE['segmentation_id'], sot.segmentation_id)