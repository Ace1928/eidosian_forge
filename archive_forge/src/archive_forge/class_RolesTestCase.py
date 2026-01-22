import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
class RolesTestCase(base.V3ClientTestCase):

    def check_role(self, role, role_ref=None):
        self.assertIsNotNone(role.id)
        self.assertIn('self', role.links)
        self.assertIn('/roles/' + role.id, role.links['self'])
        if role_ref:
            self.assertEqual(role_ref['name'], role.name)
            if hasattr(role_ref, 'domain'):
                self.assertEqual(role_ref['domain'], role.domain_id)
        else:
            self.assertIsNotNone(role.name)

    def test_create_role(self):
        role_ref = {'name': fixtures.RESOURCE_NAME_PREFIX + uuid.uuid4().hex}
        role = self.client.roles.create(**role_ref)
        self.addCleanup(self.client.roles.delete, role)
        self.check_role(role, role_ref)

    def test_create_domain_role(self):
        role_ref = {'name': fixtures.RESOURCE_NAME_PREFIX + uuid.uuid4().hex, 'domain': self.project_domain_id}
        role = self.client.roles.create(**role_ref)
        self.addCleanup(self.client.roles.delete, role)
        self.check_role(role, role_ref)

    def test_get_role(self):
        role = fixtures.Role(self.client, domain=self.project_domain_id)
        self.useFixture(role)
        role_ret = self.client.roles.get(role.id)
        self.check_role(role_ret, role.ref)

    def test_update_role_name(self):
        role = fixtures.Role(self.client, domain=self.project_domain_id)
        self.useFixture(role)
        new_name = fixtures.RESOURCE_NAME_PREFIX + uuid.uuid4().hex
        role_ret = self.client.roles.update(role.id, name=new_name)
        role.ref.update({'name': new_name})
        self.check_role(role_ret, role.ref)

    def test_update_role_domain(self):
        role = fixtures.Role(self.client)
        self.useFixture(role)
        domain = fixtures.Domain(self.client)
        self.useFixture(domain)
        new_domain = domain.id
        role_ret = self.client.roles.update(role.id, domain=new_domain)
        role.ref.update({'domain': new_domain})
        self.check_role(role_ret, role.ref)

    def test_list_roles_invalid_params(self):
        user = fixtures.User(self.client, self.project_domain_id)
        self.useFixture(user)
        self.assertRaises(exceptions.ValidationError, self.client.roles.list, user=user.id)
        group = fixtures.Group(self.client, self.project_domain_id)
        self.useFixture(group)
        self.assertRaises(exceptions.ValidationError, self.client.roles.list, group=group.id)

    def test_list_roles(self):
        global_role = fixtures.Role(self.client)
        self.useFixture(global_role)
        domain = fixtures.Domain(self.client)
        self.useFixture(domain)
        domain_role = fixtures.Role(self.client, domain=domain.id)
        self.useFixture(domain_role)
        global_roles = self.client.roles.list()
        domain_roles = self.client.roles.list(domain_id=domain.id)
        roles = global_roles + domain_roles
        for role in roles:
            self.check_role(role)
        self.assertIn(global_role.entity, global_roles)
        self.assertIn(domain_role.entity, domain_roles)

    def test_delete_role(self):
        role = self.client.roles.create(name=uuid.uuid4().hex, domain=self.project_domain_id)
        self.client.roles.delete(role.id)
        self.assertRaises(http.NotFound, self.client.roles.get, role.id)

    def test_grant_role_invalid_params(self):
        user = fixtures.User(self.client, self.project_domain_id)
        self.useFixture(user)
        role = fixtures.Role(self.client, domain=self.project_domain_id)
        self.useFixture(role)
        self.assertRaises(exceptions.ValidationError, self.client.roles.grant, role.id, user=user.id)
        group = fixtures.Group(self.client, self.project_domain_id)
        self.useFixture(group)
        self.assertRaises(exceptions.ValidationError, self.client.roles.grant, role.id, group=group.id)

    def test_user_domain_grant_and_revoke(self):
        user = fixtures.User(self.client, self.project_domain_id)
        self.useFixture(user)
        domain = fixtures.Domain(self.client)
        self.useFixture(domain)
        role = fixtures.Role(self.client, domain=self.project_domain_id)
        self.useFixture(role)
        self.client.roles.grant(role, user=user.id, domain=domain.id)
        roles_after_grant = self.client.roles.list(user=user.id, domain=domain.id)
        self.assertCountEqual(roles_after_grant, [role.entity])
        self.client.roles.revoke(role, user=user.id, domain=domain.id)
        roles_after_revoke = self.client.roles.list(user=user.id, domain=domain.id)
        self.assertEqual(roles_after_revoke, [])

    def test_user_project_grant_and_revoke(self):
        user = fixtures.User(self.client, self.project_domain_id)
        self.useFixture(user)
        project = fixtures.Project(self.client, self.project_domain_id)
        self.useFixture(project)
        role = fixtures.Role(self.client, domain=self.project_domain_id)
        self.useFixture(role)
        self.client.roles.grant(role, user=user.id, project=project.id)
        roles_after_grant = self.client.roles.list(user=user.id, project=project.id)
        self.assertCountEqual(roles_after_grant, [role.entity])
        self.client.roles.revoke(role, user=user.id, project=project.id)
        roles_after_revoke = self.client.roles.list(user=user.id, project=project.id)
        self.assertEqual(roles_after_revoke, [])

    def test_group_domain_grant_and_revoke(self):
        group = fixtures.Group(self.client, self.project_domain_id)
        self.useFixture(group)
        domain = fixtures.Domain(self.client)
        self.useFixture(domain)
        role = fixtures.Role(self.client, domain=self.project_domain_id)
        self.useFixture(role)
        self.client.roles.grant(role, group=group.id, domain=domain.id)
        roles_after_grant = self.client.roles.list(group=group.id, domain=domain.id)
        self.assertCountEqual(roles_after_grant, [role.entity])
        self.client.roles.revoke(role, group=group.id, domain=domain.id)
        roles_after_revoke = self.client.roles.list(group=group.id, domain=domain.id)
        self.assertEqual(roles_after_revoke, [])

    def test_group_project_grant_and_revoke(self):
        group = fixtures.Group(self.client, self.project_domain_id)
        self.useFixture(group)
        project = fixtures.Project(self.client, self.project_domain_id)
        self.useFixture(project)
        role = fixtures.Role(self.client, domain=self.project_domain_id)
        self.useFixture(role)
        self.client.roles.grant(role, group=group.id, project=project.id)
        roles_after_grant = self.client.roles.list(group=group.id, project=project.id)
        self.assertCountEqual(roles_after_grant, [role.entity])
        self.client.roles.revoke(role, group=group.id, project=project.id)
        roles_after_revoke = self.client.roles.list(group=group.id, project=project.id)
        self.assertEqual(roles_after_revoke, [])