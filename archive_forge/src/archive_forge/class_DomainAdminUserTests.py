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
class DomainAdminUserTests(base_classes.TestCaseWithBootstrap, common_auth.AuthTestMixin, _DomainUserTagTests):

    def setUp(self):
        super(DomainAdminUserTests, self).setUp()
        self.loadapp()
        self.policy_file = self.useFixture(temporaryfile.SecureTempFile())
        self.policy_file_name = self.policy_file.file_name
        self.useFixture(ksfixtures.Policy(self.config_fixture, policy_file=self.policy_file_name))
        _override_policy(self.policy_file_name)
        self.config_fixture.config(group='oslo_policy', enforce_scope=True)
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        self.domain_id = domain['id']
        domain_admin = unit.new_user_ref(domain_id=self.domain_id)
        self.user_id = PROVIDERS.identity_api.create_user(domain_admin)['id']
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.admin_role_id, user_id=self.user_id, domain_id=self.domain_id)
        auth = self.build_authentication_request(user_id=self.user_id, password=domain_admin['password'], domain_id=self.domain_id)
        with self.test_client() as c:
            r = c.post('/v3/auth/tokens', json=auth)
            self.token_id = r.headers['X-Subject-Token']
            self.headers = {'X-Auth-Token': self.token_id}

    def test_user_can_create_project_tag_in_domain(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id))
        tag = uuid.uuid4().hex
        with self.test_client() as c:
            c.put('/v3/projects/%s/tags/%s' % (project['id'], tag), headers=self.headers, expected_status_code=http.client.CREATED)

    def test_user_can_update_project_tag_in_domain(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id))
        tag = uuid.uuid4().hex
        PROVIDERS.resource_api.create_project_tag(project['id'], tag)
        new_tag = uuid.uuid4().hex
        update = {'tags': [new_tag]}
        with self.test_client() as c:
            r = c.put('/v3/projects/%s/tags' % project['id'], headers=self.headers, json=update, expected_status_code=http.client.OK)
            self.assertTrue(len(r.json['tags']) == 1)
            self.assertEqual(new_tag, r.json['tags'][0])

    def test_user_can_delete_project_tag_in_domain(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=self.domain_id))
        tag = uuid.uuid4().hex
        PROVIDERS.resource_api.create_project_tag(project['id'], tag)
        with self.test_client() as c:
            c.delete('/v3/projects/%s/tags/%s' % (project['id'], tag), headers=self.headers)