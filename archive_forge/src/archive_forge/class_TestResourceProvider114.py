import operator
import uuid
from osc_placement.tests.functional import base
class TestResourceProvider114(base.BaseTestCase):
    VERSION = '1.14'

    def test_resource_provider_create(self):
        created = self.resource_provider_create()
        self.assertIn('root_provider_uuid', created)
        self.assertIn('parent_provider_uuid', created)

    def test_resource_provider_set(self):
        created = self.resource_provider_create()
        updated = self.resource_provider_set(created['uuid'], name='some_new_name')
        self.assertIn('root_provider_uuid', updated)
        self.assertIn('parent_provider_uuid', updated)

    def test_resource_provider_show(self):
        created = self.resource_provider_create()
        retrieved = self.resource_provider_show(created['uuid'])
        self.assertIn('root_provider_uuid', retrieved)
        self.assertIn('parent_provider_uuid', retrieved)

    def test_resource_provider_list(self):
        self.resource_provider_create()
        retrieved = self.resource_provider_list()[0]
        self.assertIn('root_provider_uuid', retrieved)
        self.assertIn('parent_provider_uuid', retrieved)

    def test_resource_provider_create_with_parent(self):
        parent = self.resource_provider_create()
        child = self.resource_provider_create(parent_provider_uuid=parent['uuid'])
        self.assertEqual(child['parent_provider_uuid'], parent['uuid'])

    def test_resource_provider_create_then_set_parent(self):
        parent = self.resource_provider_create()
        wannabe_child = self.resource_provider_create()
        child = self.resource_provider_set(wannabe_child['uuid'], name='mandatory_name_1', parent_provider_uuid=parent['uuid'])
        self.assertEqual(child['parent_provider_uuid'], parent['uuid'])

    def test_resource_provider_set_reparent(self):
        parent1 = self.resource_provider_create()
        parent2 = self.resource_provider_create()
        child = self.resource_provider_create(parent_provider_uuid=parent1['uuid'])
        exc = self.assertRaises(base.CommandException, self.resource_provider_set, child['uuid'], name='mandatory_name_2', parent_provider_uuid=parent2['uuid'])
        self.assertIn('HTTP 400', str(exc))

    def test_resource_provider_list_in_tree(self):
        rp1 = self.resource_provider_create()
        rp2 = self.resource_provider_create(parent_provider_uuid=rp1['uuid'])
        rp3 = self.resource_provider_create(parent_provider_uuid=rp1['uuid'])
        self.resource_provider_create()
        retrieved = self.resource_provider_list(in_tree=rp2['uuid'])
        self.assertEqual(set([rp['uuid'] for rp in retrieved]), set([rp1['uuid'], rp2['uuid'], rp3['uuid']]))

    def test_resource_provider_delete_parent(self):
        parent = self.resource_provider_create()
        self.resource_provider_create(parent_provider_uuid=parent['uuid'])
        exc = self.assertRaises(base.CommandException, self.resource_provider_delete, parent['uuid'])
        self.assertIn('HTTP 409', str(exc))