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
class LDAPMatchingRuleInChainTests(LDAPTestSetup, unit.TestCase):

    def setUp(self):
        super(LDAPMatchingRuleInChainTests, self).setUp()
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        self.group = PROVIDERS.identity_api.create_group(group)
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        self.user = PROVIDERS.identity_api.create_user(user)
        PROVIDERS.identity_api.add_user_to_group(self.user['id'], self.group['id'])

    def assert_backends(self):
        _assert_backends(self, identity='ldap')

    def config_overrides(self):
        super(LDAPMatchingRuleInChainTests, self).config_overrides()
        self.config_fixture.config(group='identity', driver='ldap')
        self.config_fixture.config(group='ldap', group_ad_nesting=True, url='fake://memory', chase_referrals=False, group_tree_dn='cn=UserGroups,cn=example,cn=com', query_scope='one')

    def config_files(self):
        config_files = super(LDAPMatchingRuleInChainTests, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_ldap.conf'))
        return config_files

    def test_get_group(self):
        group_ref = PROVIDERS.identity_api.get_group(self.group['id'])
        self.assertDictEqual(self.group, group_ref)

    def test_list_user_groups(self):
        PROVIDERS.identity_api.list_groups_for_user(self.user['id'])

    def test_list_groups_for_user(self):
        groups_ref = PROVIDERS.identity_api.list_groups_for_user(self.user['id'])
        self.assertEqual(0, len(groups_ref))

    def test_list_groups(self):
        groups_refs = PROVIDERS.identity_api.list_groups()
        self.assertEqual(1, len(groups_refs))
        self.assertEqual(self.group['id'], groups_refs[0]['id'])