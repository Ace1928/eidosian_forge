import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _SystemDomainAndProjectUserDomainConfigTests(object):

    def test_user_can_get_security_compliance_domain_config(self):
        password_regex = uuid.uuid4().hex
        password_regex_description = uuid.uuid4().hex
        self.config_fixture.config(group='security_compliance', password_regex=password_regex)
        self.config_fixture.config(group='security_compliance', password_regex_description=password_regex_description)
        with self.test_client() as c:
            c.get('/v3/domains/%s/config/security_compliance' % CONF.identity.default_domain_id, headers=self.headers)

    def test_user_can_get_security_compliance_domain_config_option(self):
        password_regex_description = uuid.uuid4().hex
        self.config_fixture.config(group='security_compliance', password_regex_description=password_regex_description)
        with self.test_client() as c:
            c.get('/v3/domains/%s/config/security_compliance/password_regex_description' % CONF.identity.default_domain_id, headers=self.headers)

    def test_can_get_security_compliance_config_with_user_from_other_domain(self):
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user = unit.create_user(PROVIDERS.identity_api, domain['id'])
        project = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project['id'], project)
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user['id'], project['id'], role['id'])
        password_regex = uuid.uuid4().hex
        password_regex_description = uuid.uuid4().hex
        group = 'security_compliance'
        self.config_fixture.config(group=group, password_regex=password_regex)
        self.config_fixture.config(group=group, password_regex_description=password_regex_description)
        with self.test_client() as c:
            c.get('/v3/domains/%s/config/security_compliance' % CONF.identity.default_domain_id, headers=self.headers)