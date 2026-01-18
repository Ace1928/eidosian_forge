import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
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