import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_update_project_domain_not_allowed(self):
    domain = fixtures.Domain(self.client)
    self.useFixture(domain)
    self.assertRaises(http.BadRequest, self.client.projects.update, self.test_project.id, domain=domain.id)