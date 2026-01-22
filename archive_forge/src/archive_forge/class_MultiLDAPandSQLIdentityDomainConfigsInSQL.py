import copy
from unittest import mock
import uuid
import fixtures
import http.client
import ldap
from oslo_log import versionutils
import pkg_resources
from testtools import matchers
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import identity
from keystone.identity.backends import ldap as ldap_identity
from keystone.identity.backends.ldap import common as common_ldap
from keystone.identity.backends import sql as sql_identity
from keystone.identity.mapping_backends import mapping as map
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
from keystone.tests.unit.resource import test_backends as resource_tests
class MultiLDAPandSQLIdentityDomainConfigsInSQL(MultiLDAPandSQLIdentity):
    """Class to test the use of domain configs stored in the database.

    Repeat the same tests as MultiLDAPandSQLIdentity, but instead of using the
    domain specific config files, store the domain specific values in the
    database.

    """

    def assert_backends(self):
        _assert_backends(self, assignment='sql', identity={None: 'sql', self.domain_default['id']: 'ldap', self.domains['domain1']['id']: 'ldap', self.domains['domain2']['id']: 'ldap'}, resource='sql')

    def enable_multi_domain(self):
        default_config = {'ldap': {'url': 'fake://memory', 'user': 'cn=Admin', 'password': 'password', 'suffix': 'cn=example,cn=com'}, 'identity': {'driver': 'ldap'}}
        domain1_config = {'ldap': {'url': 'fake://memory1', 'user': 'cn=Admin', 'password': 'password', 'suffix': 'cn=example,cn=com'}, 'identity': {'driver': 'ldap', 'list_limit': 101}}
        domain2_config = {'ldap': {'url': 'fake://memory', 'user': 'cn=Admin', 'password': 'password', 'suffix': 'cn=myroot,cn=com', 'group_tree_dn': 'ou=UserGroups,dc=myroot,dc=org', 'user_tree_dn': 'ou=Users,dc=myroot,dc=org'}, 'identity': {'driver': 'ldap'}}
        PROVIDERS.domain_config_api.create_config(CONF.identity.default_domain_id, default_config)
        PROVIDERS.domain_config_api.create_config(self.domains['domain1']['id'], domain1_config)
        PROVIDERS.domain_config_api.create_config(self.domains['domain2']['id'], domain2_config)
        self.config_fixture.config(group='identity', domain_specific_drivers_enabled=True, domain_configurations_from_database=True, list_limit=1000)
        self.config_fixture.config(group='identity_mapping', backward_compatible_ids=False)

    def test_domain_config_has_no_impact_if_database_support_disabled(self):
        """Ensure database domain configs have no effect if disabled.

        Set reading from database configs to false, restart the backends
        and then try and set and use database configs.

        """
        self.config_fixture.config(group='identity', domain_configurations_from_database=False)
        self.load_backends()
        new_config = {'ldap': {'url': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(CONF.identity.default_domain_id, new_config)
        PROVIDERS.identity_api.list_users()
        default_config = PROVIDERS.identity_api.domain_configs.get_domain_conf(CONF.identity.default_domain_id)
        self.assertEqual(CONF.ldap.url, default_config.ldap.url)

    def test_reloading_domain_config(self):
        """Ensure domain drivers are reloaded on a config modification."""
        domain_cfgs = PROVIDERS.identity_api.domain_configs
        new_config = {'ldap': {'url': uuid.uuid4().hex}, 'identity': {'driver': 'ldap'}}
        PROVIDERS.domain_config_api.create_config(CONF.identity.default_domain_id, new_config)
        default_config = domain_cfgs.get_domain_conf(CONF.identity.default_domain_id)
        self.assertEqual(new_config['ldap']['url'], default_config.ldap.url)
        updated_config = {'url': uuid.uuid4().hex}
        PROVIDERS.domain_config_api.update_config(CONF.identity.default_domain_id, updated_config, group='ldap', option='url')
        default_config = domain_cfgs.get_domain_conf(CONF.identity.default_domain_id)
        self.assertEqual(updated_config['url'], default_config.ldap.url)
        PROVIDERS.domain_config_api.delete_config(CONF.identity.default_domain_id)
        default_config = domain_cfgs.get_domain_conf(CONF.identity.default_domain_id)
        self.assertEqual(CONF.ldap.url, default_config.ldap.url)

    def test_setting_multiple_sql_driver_raises_exception(self):
        """Ensure setting multiple domain specific sql drivers is prevented."""
        new_config = {'identity': {'driver': 'sql'}}
        PROVIDERS.domain_config_api.create_config(CONF.identity.default_domain_id, new_config)
        PROVIDERS.identity_api.domain_configs.get_domain_conf(CONF.identity.default_domain_id)
        PROVIDERS.domain_config_api.create_config(self.domains['domain1']['id'], new_config)
        self.assertRaises(exception.MultipleSQLDriversInConfig, PROVIDERS.identity_api.domain_configs.get_domain_conf, self.domains['domain1']['id'])

    def test_same_domain_gets_sql_driver(self):
        """Ensure we can set an SQL driver if we have had it before."""
        new_config = {'identity': {'driver': 'sql'}}
        PROVIDERS.domain_config_api.create_config(CONF.identity.default_domain_id, new_config)
        PROVIDERS.identity_api.domain_configs.get_domain_conf(CONF.identity.default_domain_id)
        new_config = {'identity': {'driver': 'sql'}, 'ldap': {'url': 'fake://memory1'}}
        PROVIDERS.domain_config_api.create_config(CONF.identity.default_domain_id, new_config)
        PROVIDERS.identity_api.domain_configs.get_domain_conf(CONF.identity.default_domain_id)

    def test_delete_domain_clears_sql_registration(self):
        """Ensure registration is deleted when a domain is deleted."""
        domain = unit.new_domain_ref()
        domain = PROVIDERS.resource_api.create_domain(domain['id'], domain)
        new_config = {'identity': {'driver': 'sql'}}
        PROVIDERS.domain_config_api.create_config(domain['id'], new_config)
        PROVIDERS.identity_api.domain_configs.get_domain_conf(domain['id'])
        PROVIDERS.domain_config_api.create_config(self.domains['domain1']['id'], new_config)
        self.assertRaises(exception.MultipleSQLDriversInConfig, PROVIDERS.identity_api.domain_configs.get_domain_conf, self.domains['domain1']['id'])
        PROVIDERS.domain_config_api.delete_config(self.domains['domain1']['id'])
        domain['enabled'] = False
        PROVIDERS.resource_api.update_domain(domain['id'], domain)
        PROVIDERS.resource_api.delete_domain(domain['id'])
        PROVIDERS.domain_config_api.create_config(self.domains['domain1']['id'], new_config)
        PROVIDERS.identity_api.domain_configs.get_domain_conf(self.domains['domain1']['id'])

    def test_orphaned_registration_does_not_prevent_getting_sql_driver(self):
        """Ensure we self heal an orphaned sql registration."""
        domain = unit.new_domain_ref()
        domain = PROVIDERS.resource_api.create_domain(domain['id'], domain)
        new_config = {'identity': {'driver': 'sql'}}
        PROVIDERS.domain_config_api.create_config(domain['id'], new_config)
        PROVIDERS.identity_api.domain_configs.get_domain_conf(domain['id'])
        PROVIDERS.domain_config_api.create_config(self.domains['domain1']['id'], new_config)
        self.assertRaises(exception.MultipleSQLDriversInConfig, PROVIDERS.identity_api.domain_configs.get_domain_conf, self.domains['domain1']['id'])
        PROVIDERS.resource_api.driver.delete_project(domain['id'])
        PROVIDERS.resource_api.get_domain.invalidate(PROVIDERS.resource_api, domain['id'])
        PROVIDERS.domain_config_api.create_config(self.domains['domain1']['id'], new_config)
        PROVIDERS.identity_api.domain_configs.get_domain_conf(self.domains['domain1']['id'])