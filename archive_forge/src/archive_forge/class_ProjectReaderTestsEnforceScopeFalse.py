import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as bp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class ProjectReaderTestsEnforceScopeFalse(base_classes.TestCaseWithBootstrap, common_auth.AuthTestMixin, _UserCredentialTests, _ProjectUsersTests):

    def setUp(self):
        super(ProjectReaderTestsEnforceScopeFalse, self).setUp()
        self.loadapp()
        self.useFixture(ksfixtures.Policy(self.config_fixture))
        self.config_fixture.config(group='oslo_policy', enforce_scope=False)
        project_reader = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        self.user_id = PROVIDERS.identity_api.create_user(project_reader)['id']
        project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        self.project_id = PROVIDERS.resource_api.create_project(project['id'], project)['id']
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=self.user_id, project_id=self.project_id)
        auth = self.build_authentication_request(user_id=self.user_id, password=project_reader['password'], project_id=self.project_id)
        with self.test_client() as c:
            r = c.post('/v3/auth/tokens', json=auth)
            self.token_id = r.headers['X-Subject-Token']
            self.headers = {'X-Auth-Token': self.token_id}