import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _SystemUserDomainConfigTests(object):

    def test_user_can_get_domain_config(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.domain_config_api.create_config(domain['id'], unit.new_domain_config_ref())
        with self.test_client() as c:
            c.get('/v3/domains/%s/config' % domain['id'], headers=self.headers)

    def test_user_can_get_domain_group_config(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.domain_config_api.create_config(domain['id'], unit.new_domain_config_ref())
        with self.test_client() as c:
            c.get('/v3/domains/%s/config/ldap' % domain['id'], headers=self.headers)

    def test_user_can_get_config_by_group_invalid_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.domain_config_api.create_config(domain['id'], unit.new_domain_config_ref())
        invalid_domain_id = uuid.uuid4().hex
        with self.test_client() as c:
            c.get('/v3/domains/%s/config/ldap' % invalid_domain_id, headers=self.headers, expected_status_code=http.client.NOT_FOUND)

    def test_user_can_get_non_existent_config(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        with self.test_client() as c:
            c.get('/v3/domains/%s/config' % domain['id'], headers=self.headers, expected_status_code=http.client.NOT_FOUND)

    def test_user_can_get_non_existent_config_group_invalid_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        config = {'ldap': {'url': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(domain['id'], config)
        invalid_domain_id = uuid.uuid4().hex
        with self.test_client() as c:
            c.get('/v3/domains/%s/config/ldap' % invalid_domain_id, headers=self.headers, expected_status_code=http.client.NOT_FOUND)

    def test_user_can_get_domain_config_option(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        PROVIDERS.domain_config_api.create_config(domain['id'], unit.new_domain_config_ref())
        with self.test_client() as c:
            c.get('/v3/domains/%s/config/ldap/url' % domain['id'], headers=self.headers)

    def test_user_can_get_non_existent_config_option(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        config = {'ldap': {'url': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(domain['id'], config)
        with self.test_client() as c:
            c.get('/v3/domains/%s/config/ldap/user_tree_dn' % domain['id'], headers=self.headers, expected_status_code=http.client.NOT_FOUND)

    def test_user_can_get_non_existent_config_option_invalid_domain(self):
        domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
        config = {'ldap': {'url': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(domain['id'], config)
        invalid_domain_id = uuid.uuid4().hex
        with self.test_client() as c:
            c.get('/v3/domains/%s/config/ldap/user_tree_dn' % invalid_domain_id, headers=self.headers, expected_status_code=http.client.NOT_FOUND)

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
        password_regex = uuid.uuid4().hex
        password_regex_description = uuid.uuid4().hex
        group = 'security_compliance'
        self.config_fixture.config(group=group, password_regex=password_regex)
        self.config_fixture.config(group=group, password_regex_description=password_regex_description)
        with self.test_client() as c:
            c.get('/v3/domains/%s/config/security_compliance' % CONF.identity.default_domain_id, headers=self.headers)

    def test_user_can_get_domain_config_default(self):
        with self.test_client() as c:
            c.get('/v3/domains/config/default', headers=self.headers)

    def test_user_can_get_domain_group_config_default(self):
        with self.test_client() as c:
            c.get('/v3/domains/config/ldap/default', headers=self.headers)

    def test_user_can_get_domain_config_option_default(self):
        with self.test_client() as c:
            c.get('/v3/domains/config/ldap/url/default', headers=self.headers)