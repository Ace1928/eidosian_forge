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
class LDAPIdentityEnabledEmulation(LDAPIdentity, unit.TestCase):

    def setUp(self):
        super(LDAPIdentityEnabledEmulation, self).setUp()
        _assert_backends(self, identity='ldap')

    def load_fixtures(self, fixtures):
        super(LDAPIdentity, self).load_fixtures(fixtures)
        for obj in [self.project_bar, self.project_baz, self.user_foo, self.user_two, self.user_badguy]:
            obj.setdefault('enabled', True)

    def config_files(self):
        config_files = super(LDAPIdentityEnabledEmulation, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_ldap.conf'))
        return config_files

    def config_overrides(self):
        super(LDAPIdentityEnabledEmulation, self).config_overrides()
        self.config_fixture.config(group='ldap', user_enabled_emulation=True)

    def test_project_crud(self):
        project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        project = PROVIDERS.resource_api.create_project(project['id'], project)
        project_ref = PROVIDERS.resource_api.get_project(project['id'])
        project['enabled'] = True
        self.assertDictEqual(project, project_ref)
        project['description'] = uuid.uuid4().hex
        PROVIDERS.resource_api.update_project(project['id'], project)
        project_ref = PROVIDERS.resource_api.get_project(project['id'])
        self.assertDictEqual(project, project_ref)
        PROVIDERS.resource_api.delete_project(project['id'])
        self.assertRaises(exception.ProjectNotFound, PROVIDERS.resource_api.get_project, project['id'])

    def test_user_auth_emulated(self):
        driver = PROVIDERS.identity_api._select_identity_driver(CONF.identity.default_domain_id)
        driver.user.enabled_emulation_dn = 'cn=test,dc=test'
        with self.make_request():
            PROVIDERS.identity_api.authenticate(user_id=self.user_foo['id'], password=self.user_foo['password'])

    def test_user_enable_attribute_mask(self):
        self.skip_test_overrides('Enabled emulation conflicts with enabled mask')

    def test_user_enabled_use_group_config(self):
        group_name = 'enabled_users'
        driver = PROVIDERS.identity_api._select_identity_driver(CONF.identity.default_domain_id)
        group_dn = 'cn=%s,%s' % (group_name, driver.group.tree_dn)
        self.config_fixture.config(group='ldap', user_enabled_emulation_use_group_config=True, user_enabled_emulation_dn=group_dn, group_name_attribute='cn', group_member_attribute='uniqueMember', group_objectclass='groupOfUniqueNames')
        self.ldapdb.clear()
        self.load_backends()
        user1 = unit.new_user_ref(enabled=True, domain_id=CONF.identity.default_domain_id)
        user_ref = PROVIDERS.identity_api.create_user(user1)
        self.assertIs(True, user_ref['enabled'])
        user_ref = PROVIDERS.identity_api.get_user(user_ref['id'])
        self.assertIs(True, user_ref['enabled'])
        group_ref = PROVIDERS.identity_api.get_group_by_name(group_name, CONF.identity.default_domain_id)
        PROVIDERS.identity_api.check_user_in_group(user_ref['id'], group_ref['id'])

    def test_user_enabled_use_group_config_with_ids(self):
        group_name = 'enabled_users'
        driver = PROVIDERS.identity_api._select_identity_driver(CONF.identity.default_domain_id)
        group_dn = 'cn=%s,%s' % (group_name, driver.group.tree_dn)
        self.config_fixture.config(group='ldap', user_enabled_emulation_use_group_config=True, user_enabled_emulation_dn=group_dn, group_name_attribute='cn', group_member_attribute='memberUid', group_members_are_ids=True, group_objectclass='posixGroup')
        self.ldapdb.clear()
        self.load_backends()
        user1 = unit.new_user_ref(enabled=True, domain_id=CONF.identity.default_domain_id)
        user_ref = PROVIDERS.identity_api.create_user(user1)
        self.assertIs(True, user_ref['enabled'])
        user_ref = PROVIDERS.identity_api.get_user(user_ref['id'])
        self.assertIs(True, user_ref['enabled'])
        group_ref = PROVIDERS.identity_api.get_group_by_name(group_name, CONF.identity.default_domain_id)
        PROVIDERS.identity_api.check_user_in_group(user_ref['id'], group_ref['id'])

    def test_user_enabled_invert(self):
        self.config_fixture.config(group='ldap', user_enabled_invert=True, user_enabled_default='False')
        self.ldapdb.clear()
        self.load_backends()
        user1 = self.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user2 = self.new_user_ref(enabled=False, domain_id=CONF.identity.default_domain_id)
        user3 = self.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_ref = PROVIDERS.identity_api.create_user(user1)
        self.assertIs(True, user_ref['enabled'])
        self.assertIsNone(self.get_user_enabled_vals(user_ref))
        user_ref = PROVIDERS.identity_api.get_user(user_ref['id'])
        self.assertIs(True, user_ref['enabled'])
        user1['enabled'] = False
        user_ref = PROVIDERS.identity_api.update_user(user_ref['id'], user1)
        self.assertIs(False, user_ref['enabled'])
        self.assertIsNone(self.get_user_enabled_vals(user_ref))
        user1['enabled'] = True
        user_ref = PROVIDERS.identity_api.update_user(user_ref['id'], user1)
        self.assertIs(True, user_ref['enabled'])
        self.assertIsNone(self.get_user_enabled_vals(user_ref))
        user_ref = PROVIDERS.identity_api.create_user(user2)
        self.assertIs(False, user_ref['enabled'])
        self.assertIsNone(self.get_user_enabled_vals(user_ref))
        user_ref = PROVIDERS.identity_api.get_user(user_ref['id'])
        self.assertIs(False, user_ref['enabled'])
        user_ref = PROVIDERS.identity_api.create_user(user3)
        self.assertIs(True, user_ref['enabled'])
        self.assertIsNone(self.get_user_enabled_vals(user_ref))
        user_ref = PROVIDERS.identity_api.get_user(user_ref['id'])
        self.assertIs(True, user_ref['enabled'])

    def test_user_enabled_invert_default_str_value(self):
        self.skip_test_overrides('N/A: Covered by test_user_enabled_invert')

    @mock.patch.object(common_ldap.BaseLdap, '_ldap_get')
    def test_user_enabled_attribute_handles_utf8(self, mock_ldap_get):
        self.config_fixture.config(group='ldap', user_enabled_invert=True, user_enabled_attribute='passwordisexpired')
        mock_ldap_get.return_value = (u'uid=123456789,c=us,ou=our_ldap,o=acme.com', {'uid': [123456789], 'mail': [u'shaun@acme.com'], 'passwordisexpired': [u'false'], 'cn': [u'uid=123456789,c=us,ou=our_ldap,o=acme.com']})
        user_api = identity.backends.ldap.UserApi(CONF)
        user_ref = user_api.get('123456789')
        self.assertIs(False, user_ref['enabled'])

    def test_escape_member_dn(self):
        object_id = uuid.uuid4().hex
        driver = PROVIDERS.identity_api._select_identity_driver(CONF.identity.default_domain_id)
        mixin_impl = driver.user
        sample_dn = 'cn=foo)bar'
        sample_dn_filter_esc = 'cn=foo\\29bar'
        mixin_impl.tree_dn = sample_dn
        exp_filter = '(%s=%s=%s,%s)' % (mixin_impl.member_attribute, mixin_impl.id_attr, object_id, sample_dn_filter_esc)
        with mixin_impl.get_connection() as conn:
            m = self.useFixture(fixtures.MockPatchObject(conn, 'search_s')).mock
            mixin_impl._is_id_enabled(object_id, conn)
            self.assertEqual(exp_filter, m.call_args[0][2])