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
class MultiLDAPandSQLIdentity(BaseLDAPIdentity, unit.SQLDriverOverrides, unit.TestCase, BaseMultiLDAPandSQLIdentity):
    """Class to test common SQL plus individual LDAP backends.

    We define a set of domains and domain-specific backends:

    - A separate LDAP backend for the default domain
    - A separate LDAP backend for domain1
    - domain2 shares the same LDAP as domain1, but uses a different
      tree attach point
    - An SQL backend for all other domains (which will include domain3
      and domain4)

    Normally one would expect that the default domain would be handled as
    part of the "other domains" - however the above provides better
    test coverage since most of the existing backend tests use the default
    domain.

    """

    def load_fixtures(self, fixtures):
        self.domain_count = 5
        self.domain_specific_count = 3
        PROVIDERS.resource_api.create_domain(default_fixtures.ROOT_DOMAIN['id'], default_fixtures.ROOT_DOMAIN)
        self.setup_initial_domains()
        self.enable_multi_domain()
        super(MultiLDAPandSQLIdentity, self).load_fixtures(fixtures)

    def assert_backends(self):
        _assert_backends(self, assignment='sql', identity={None: 'sql', self.domain_default['id']: 'ldap', self.domains['domain1']['id']: 'ldap', self.domains['domain2']['id']: 'ldap'}, resource='sql')

    def config_overrides(self):
        super(MultiLDAPandSQLIdentity, self).config_overrides()
        self.config_fixture.config(group='identity', driver='sql')
        self.config_fixture.config(group='resource', driver='sql')
        self.config_fixture.config(group='assignment', driver='sql')

    def enable_multi_domain(self):
        """Enable the chosen form of multi domain configuration support.

        This method enables the file-based configuration support. Child classes
        that wish to use the database domain configuration support should
        override this method and set the appropriate config_fixture option.

        """
        self.config_fixture.config(group='identity', domain_specific_drivers_enabled=True, domain_config_dir=unit.TESTCONF + '/domain_configs_multi_ldap', list_limit=1000)
        self.config_fixture.config(group='identity_mapping', backward_compatible_ids=False)

    def get_config(self, domain_id):
        return PROVIDERS.identity_api.domain_configs.get_domain_conf(domain_id)

    def test_list_users(self):
        _users = self.create_users_across_domains()
        users = PROVIDERS.identity_api.list_users(domain_scope=self._set_domain_scope(CONF.identity.default_domain_id))
        self.assertEqual(len(default_fixtures.USERS) + 1, len(users))
        user_ids = set((user['id'] for user in users))
        expected_user_ids = set((getattr(self, 'user_%s' % user['name'])['id'] for user in default_fixtures.USERS))
        expected_user_ids.add(_users['user0']['id'])
        for user_ref in users:
            self.assertNotIn('password', user_ref)
        self.assertEqual(expected_user_ids, user_ids)

    @mock.patch.object(common_ldap.BaseLdap, '_ldap_get_all')
    def test_list_limit_domain_specific_inheritance(self, ldap_get_all):
        hints = driver_hints.Hints()
        PROVIDERS.identity_api.list_users(domain_scope=self.domains['domain2']['id'], hints=hints)
        self.assertTrue(ldap_get_all.called)
        args, kwargs = ldap_get_all.call_args
        hints = args[0]
        self.assertEqual(1000, hints.limit['limit'])

    @mock.patch.object(common_ldap.BaseLdap, '_ldap_get_all')
    def test_list_limit_domain_specific_override(self, ldap_get_all):
        hints = driver_hints.Hints()
        PROVIDERS.identity_api.list_users(domain_scope=self.domains['domain1']['id'], hints=hints)
        self.assertTrue(ldap_get_all.called)
        args, kwargs = ldap_get_all.call_args
        hints = args[0]
        self.assertEqual(101, hints.limit['limit'])

    def test_domain_segregation(self):
        """Test that separate configs have segregated the domain.

        Test Plan:

        - Users were created in each domain as part of setup, now make sure
          you can only find a given user in its relevant domain/backend
        - Make sure that for a backend that supports multiple domains
          you can get the users via any of its domains

        """
        users = self.create_users_across_domains()
        check_user = self.check_user
        check_user(users['user0'], self.domain_default['id'], http.client.OK)
        for domain in [self.domains['domain1']['id'], self.domains['domain2']['id'], self.domains['domain3']['id'], self.domains['domain4']['id']]:
            check_user(users['user0'], domain, exception.UserNotFound)
        check_user(users['user1'], self.domains['domain1']['id'], http.client.OK)
        for domain in [self.domain_default['id'], self.domains['domain2']['id'], self.domains['domain3']['id'], self.domains['domain4']['id']]:
            check_user(users['user1'], domain, exception.UserNotFound)
        check_user(users['user2'], self.domains['domain2']['id'], http.client.OK)
        for domain in [self.domain_default['id'], self.domains['domain1']['id'], self.domains['domain3']['id'], self.domains['domain4']['id']]:
            check_user(users['user2'], domain, exception.UserNotFound)
        check_user(users['user3'], self.domains['domain3']['id'], http.client.OK)
        check_user(users['user3'], self.domains['domain4']['id'], http.client.OK)
        check_user(users['user4'], self.domains['domain3']['id'], http.client.OK)
        check_user(users['user4'], self.domains['domain4']['id'], http.client.OK)
        for domain in [self.domain_default['id'], self.domains['domain1']['id'], self.domains['domain2']['id']]:
            check_user(users['user3'], domain, exception.UserNotFound)
            check_user(users['user4'], domain, exception.UserNotFound)
        for domain in [self.domains['domain1']['id'], self.domains['domain2']['id'], self.domains['domain4']['id']]:
            self.assertThat(PROVIDERS.identity_api.list_users(domain_scope=domain), matchers.HasLength(1))
        self.assertThat(PROVIDERS.identity_api.list_users(domain_scope=self.domains['domain3']['id']), matchers.HasLength(1))

    def test_existing_uuids_work(self):
        """Test that 'uni-domain' created IDs still work.

        Throwing the switch to domain-specific backends should not cause
        existing identities to be inaccessible via ID.

        """
        userA = unit.create_user(PROVIDERS.identity_api, self.domain_default['id'])
        userB = unit.create_user(PROVIDERS.identity_api, self.domains['domain1']['id'])
        userC = unit.create_user(PROVIDERS.identity_api, self.domains['domain3']['id'])
        PROVIDERS.identity_api.get_user(userA['id'])
        PROVIDERS.identity_api.get_user(userB['id'])
        PROVIDERS.identity_api.get_user(userC['id'])

    def test_scanning_of_config_dir(self):
        """Test the Manager class scans the config directory.

        The setup for the main tests above load the domain configs directly
        so that the test overrides can be included. This test just makes sure
        that the standard config directory scanning does pick up the relevant
        domain config files.

        """
        self.assertTrue(CONF.identity.domain_specific_drivers_enabled)
        self.load_backends()
        PROVIDERS.identity_api.list_users(domain_scope=self.domains['domain1']['id'])
        self.assertIn('default', PROVIDERS.identity_api.domain_configs)
        self.assertIn(self.domains['domain1']['id'], PROVIDERS.identity_api.domain_configs)
        self.assertIn(self.domains['domain2']['id'], PROVIDERS.identity_api.domain_configs)
        self.assertNotIn(self.domains['domain3']['id'], PROVIDERS.identity_api.domain_configs)
        self.assertNotIn(self.domains['domain4']['id'], PROVIDERS.identity_api.domain_configs)
        conf = PROVIDERS.identity_api.domain_configs.get_domain_conf(self.domains['domain1']['id'])
        self.assertFalse(conf.identity.domain_specific_drivers_enabled)
        self.assertEqual('fake://memory1', conf.ldap.url)

    def test_delete_domain_with_user_added(self):
        domain = unit.new_domain_ref()
        project = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        project = PROVIDERS.resource_api.create_project(project['id'], project)
        project_ref = PROVIDERS.resource_api.get_project(project['id'])
        self.assertDictEqual(project, project_ref)
        PROVIDERS.assignment_api.create_grant(user_id=self.user_foo['id'], project_id=project['id'], role_id=self.role_member['id'])
        PROVIDERS.assignment_api.delete_grant(user_id=self.user_foo['id'], project_id=project['id'], role_id=self.role_member['id'])
        domain['enabled'] = False
        PROVIDERS.resource_api.update_domain(domain['id'], domain)
        PROVIDERS.resource_api.delete_domain(domain['id'])
        self.assertRaises(exception.DomainNotFound, PROVIDERS.resource_api.get_domain, domain['id'])

    def test_user_enabled_ignored_disable_error(self):
        self.skip_test_overrides("Doesn't apply since LDAP config has no affect on the SQL identity backend.")

    def test_group_enabled_ignored_disable_error(self):
        self.skip_test_overrides("Doesn't apply since LDAP config has no affect on the SQL identity backend.")

    def test_list_role_assignments_filtered_by_role(self):
        base = super(BaseLDAPIdentity, self)
        base.test_list_role_assignments_filtered_by_role()

    def test_list_role_assignment_by_domain(self):
        super(BaseLDAPIdentity, self).test_list_role_assignment_by_domain()

    def test_list_role_assignment_by_user_with_domain_group_roles(self):
        super(BaseLDAPIdentity, self).test_list_role_assignment_by_user_with_domain_group_roles()

    def test_list_role_assignment_using_sourced_groups_with_domains(self):
        base = super(BaseLDAPIdentity, self)
        base.test_list_role_assignment_using_sourced_groups_with_domains()

    def test_create_project_with_domain_id_and_without_parent_id(self):
        super(BaseLDAPIdentity, self).test_create_project_with_domain_id_and_without_parent_id()

    def test_create_project_with_domain_id_mismatch_to_parent_domain(self):
        super(BaseLDAPIdentity, self).test_create_project_with_domain_id_mismatch_to_parent_domain()

    def test_remove_foreign_assignments_when_deleting_a_domain(self):
        base = super(BaseLDAPIdentity, self)
        base.test_remove_foreign_assignments_when_deleting_a_domain()

    @mock.patch.object(ldap_identity.Identity, 'unset_default_project_id')
    @mock.patch.object(sql_identity.Identity, 'unset_default_project_id')
    def test_delete_project_unset_project_ids_for_all_backends(self, sql_mock, ldap_mock):
        ldap_mock.side_effect = exception.Forbidden
        project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        project = PROVIDERS.resource_api.create_project(project['id'], project)
        PROVIDERS.resource_api.delete_project(project['id'])
        ldap_mock.assert_called_with(project['id'])
        sql_mock.assert_called_with(project['id'])