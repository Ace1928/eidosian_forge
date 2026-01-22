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
class DomainReaderUserTests(base_classes.TestCaseWithBootstrap, common_auth.AuthTestMixin, _DomainUserTagTests, _DomainMemberAndReaderTagTests):

    def setUp(self):
        super(DomainReaderUserTests, self).setUp()
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
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=self.user_id, domain_id=self.domain_id)
        auth = self.build_authentication_request(user_id=self.user_id, password=domain_admin['password'], domain_id=self.domain_id)
        with self.test_client() as c:
            r = c.post('/v3/auth/tokens', json=auth)
            self.token_id = r.headers['X-Subject-Token']
            self.headers = {'X-Auth-Token': self.token_id}