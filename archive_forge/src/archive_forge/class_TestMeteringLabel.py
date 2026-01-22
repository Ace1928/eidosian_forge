from openstack.network.v2 import metering_label
from openstack.tests.unit import base
class TestMeteringLabel(base.TestCase):

    def test_basic(self):
        sot = metering_label.MeteringLabel()
        self.assertEqual('metering_label', sot.resource_key)
        self.assertEqual('metering_labels', sot.resources_key)
        self.assertEqual('/metering/metering-labels', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = metering_label.MeteringLabel(**EXAMPLE)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['shared'], sot.is_shared)