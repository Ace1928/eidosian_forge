import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import grant as gp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _SystemUserGrantTests(object):

    def test_can_list_grants_for_user_on_project(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=CONF.identity.default_domain_id))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], project_id=project['id'])
        with self.test_client() as c:
            r = c.get('/v3/projects/%s/users/%s/roles' % (project['id'], user['id']), headers=self.headers)
            self.assertEqual(1, len(r.json['roles']))

    def test_can_list_grants_for_user_on_domain(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=CONF.identity.default_domain_id))
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], domain_id=domain['id'])
        with self.test_client() as c:
            r = c.get('/v3/domains/%s/users/%s/roles' % (domain['id'], user['id']), headers=self.headers)
            self.assertEqual(1, len(r.json['roles']))

    def test_can_list_grants_for_group_on_project(self):
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=CONF.identity.default_domain_id))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, group_id=group['id'], project_id=project['id'])
        with self.test_client() as c:
            r = c.get('/v3/projects/%s/groups/%s/roles' % (project['id'], group['id']), headers=self.headers)
            self.assertEqual(1, len(r.json['roles']))

    def test_can_list_grants_for_group_on_domain(self):
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=CONF.identity.default_domain_id))
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, group_id=group['id'], domain_id=domain['id'])
        with self.test_client() as c:
            r = c.get('/v3/domains/%s/groups/%s/roles' % (domain['id'], group['id']), headers=self.headers)
            self.assertEqual(1, len(r.json['roles']))

    def test_can_check_grant_for_user_on_project(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=CONF.identity.default_domain_id))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], project_id=project['id'])
        with self.test_client() as c:
            c.get('/v3/projects/%s/users/%s/roles/%s' % (project['id'], user['id'], self.bootstrapper.reader_role_id), headers=self.headers, expected_status_code=http.client.NO_CONTENT)

    def test_can_check_grant_for_user_on_domain(self):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=CONF.identity.default_domain_id))
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], domain_id=domain['id'])
        with self.test_client() as c:
            c.get('/v3/domains/%s/users/%s/roles/%s' % (domain['id'], user['id'], self.bootstrapper.reader_role_id), headers=self.headers, expected_status_code=http.client.NO_CONTENT)

    def test_can_check_grant_for_group_on_project(self):
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=CONF.identity.default_domain_id))
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, group_id=group['id'], project_id=project['id'])
        with self.test_client() as c:
            c.get('/v3/projects/%s/groups/%s/roles/%s' % (project['id'], group['id'], self.bootstrapper.reader_role_id), headers=self.headers, expected_status_code=http.client.NO_CONTENT)

    def test_can_check_grant_for_group_on_domain(self):
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=CONF.identity.default_domain_id))
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, group_id=group['id'], domain_id=domain['id'])
        with self.test_client() as c:
            c.get('/v3/domains/%s/groups/%s/roles/%s' % (domain['id'], group['id'], self.bootstrapper.reader_role_id), headers=self.headers, expected_status_code=http.client.NO_CONTENT)