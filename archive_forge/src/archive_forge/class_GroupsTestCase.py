import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
class GroupsTestCase(base.V3ClientTestCase):

    def check_group(self, group, group_ref=None):
        self.assertIsNotNone(group.id)
        self.assertIn('self', group.links)
        self.assertIn('/groups/' + group.id, group.links['self'])
        if group_ref:
            self.assertEqual(group_ref['name'], group.name)
            self.assertEqual(group_ref['domain'], group.domain_id)
            if hasattr(group_ref, 'description'):
                self.assertEqual(group_ref['description'], group.description)
        else:
            self.assertIsNotNone(group.name)
            self.assertIsNotNone(group.domain_id)

    def test_create_group(self):
        group_ref = {'name': fixtures.RESOURCE_NAME_PREFIX + uuid.uuid4().hex, 'domain': self.project_domain_id, 'description': uuid.uuid4().hex}
        group = self.client.groups.create(**group_ref)
        self.addCleanup(self.client.groups.delete, group)
        self.check_group(group, group_ref)

    def test_get_group(self):
        group = fixtures.Group(self.client, self.project_domain_id)
        self.useFixture(group)
        group_ret = self.client.groups.get(group.id)
        self.check_group(group_ret, group.ref)

    def test_list_groups(self):
        group_one = fixtures.Group(self.client, self.project_domain_id)
        self.useFixture(group_one)
        group_two = fixtures.Group(self.client, self.project_domain_id)
        self.useFixture(group_two)
        groups = self.client.groups.list()
        for group in groups:
            self.check_group(group)
        self.assertIn(group_one.entity, groups)
        self.assertIn(group_two.entity, groups)

    def test_update_group(self):
        group = fixtures.Group(self.client, self.project_domain_id)
        self.useFixture(group)
        new_name = fixtures.RESOURCE_NAME_PREFIX + uuid.uuid4().hex
        new_description = uuid.uuid4().hex
        group_ret = self.client.groups.update(group.id, name=new_name, description=new_description)
        group.ref.update({'name': new_name, 'description': new_description})
        self.check_group(group_ret, group.ref)

    def test_delete_group(self):
        group = self.client.groups.create(name=uuid.uuid4().hex, domain=self.project_domain_id)
        self.client.groups.delete(group.id)
        self.assertRaises(http.NotFound, self.client.groups.get, group.id)