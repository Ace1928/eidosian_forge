import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import project as pp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _DomainUsersTests(object):
    """Common default functionality for all domain users."""

    def test_user_can_list_projects_within_domain(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id))
        with self.test_client() as c:
            r = c.get('/v3/projects', headers=self.headers)
            self.assertEqual(1, len(r.json['projects']))
            self.assertEqual(project['id'], r.json['projects'][0]['id'])

    def test_user_cannot_list_projects_in_other_domain(self):
        PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        with self.test_client() as c:
            r = c.get('/v3/projects', headers=self.headers)
            self.assertEqual(0, len(r.json['projects']))

    def test_user_can_get_a_project_within_domain(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id))
        with self.test_client() as c:
            r = c.get('/v3/projects/%s' % project['id'], headers=self.headers)
            self.assertEqual(project['id'], r.json['project']['id'])

    def test_user_cannot_get_a_project_in_other_domain(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        with self.test_client() as c:
            c.get('/v3/projects/%s' % project['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_can_list_projects_for_user_in_domain(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(self.domain_id, id=uuid.uuid4().hex))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id))
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], project_id=project['id'])
        with self.test_client() as c:
            r = c.get('/v3/users/%s/projects' % user['id'], headers=self.headers)
            self.assertEqual(1, len(r.json['projects']))
            self.assertEqual(project['id'], r.json['projects'][0]['id'])

    def test_user_cannot_list_projects_for_user_in_other_domain(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(CONF.identity.default_domain_id, id=uuid.uuid4().hex))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], project_id=project['id'])
        with self.test_client() as c:
            c.get('/v3/users/%s/projects' % user['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)