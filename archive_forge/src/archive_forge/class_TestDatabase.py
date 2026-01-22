from openstack.database.v1 import database
from openstack.tests.unit import base
class TestDatabase(base.TestCase):

    def test_basic(self):
        sot = database.Database()
        self.assertEqual('database', sot.resource_key)
        self.assertEqual('databases', sot.resources_key)
        path = '/instances/%(instance_id)s/databases'
        self.assertEqual(path, sot.base_path)
        self.assertTrue(sot.allow_list)
        self.assertTrue(sot.allow_create)
        self.assertFalse(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertTrue(sot.allow_delete)

    def test_make_it(self):
        sot = database.Database(**EXAMPLE)
        self.assertEqual(IDENTIFIER, sot.id)
        self.assertEqual(EXAMPLE['character_set'], sot.character_set)
        self.assertEqual(EXAMPLE['collate'], sot.collate)
        self.assertEqual(EXAMPLE['instance_id'], sot.instance_id)
        self.assertEqual(IDENTIFIER, sot.name)
        self.assertEqual(IDENTIFIER, sot.id)