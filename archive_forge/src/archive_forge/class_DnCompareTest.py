import os
import tempfile
from unittest import mock
import uuid
import fixtures
import ldap.dn
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception as ks_exception
from keystone.identity.backends.ldap import common as common_ldap
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import fakeldap
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
class DnCompareTest(unit.BaseTestCase):
    """Test for the DN comparison functions in keystone.common.ldap.core."""

    def test_prep(self):
        value = 'lowercase value'
        self.assertEqual(value, common_ldap.prep_case_insensitive(value))

    def test_prep_lowercase(self):
        value = 'UPPERCASE VALUE'
        exp_value = value.lower()
        self.assertEqual(exp_value, common_ldap.prep_case_insensitive(value))

    def test_prep_insignificant(self):
        value = 'before   after'
        exp_value = 'before after'
        self.assertEqual(exp_value, common_ldap.prep_case_insensitive(value))

    def test_prep_insignificant_pre_post(self):
        value = '   value   '
        exp_value = 'value'
        self.assertEqual(exp_value, common_ldap.prep_case_insensitive(value))

    def test_ava_equal_same(self):
        value = 'val1'
        self.assertTrue(common_ldap.is_ava_value_equal('cn', value, value))

    def test_ava_equal_complex(self):
        val1 = 'before   after'
        val2 = '  BEFORE  afTer '
        self.assertTrue(common_ldap.is_ava_value_equal('cn', val1, val2))

    def test_ava_different(self):
        self.assertFalse(common_ldap.is_ava_value_equal('cn', 'val1', 'val2'))

    def test_rdn_same(self):
        rdn = ldap.dn.str2dn('cn=val1')[0]
        self.assertTrue(common_ldap.is_rdn_equal(rdn, rdn))

    def test_rdn_diff_length(self):
        rdn1 = ldap.dn.str2dn('cn=cn1')[0]
        rdn2 = ldap.dn.str2dn('cn=cn1+ou=ou1')[0]
        self.assertFalse(common_ldap.is_rdn_equal(rdn1, rdn2))

    def test_rdn_multi_ava_same_order(self):
        rdn1 = ldap.dn.str2dn('cn=cn1+ou=ou1')[0]
        rdn2 = ldap.dn.str2dn('cn=CN1+ou=OU1')[0]
        self.assertTrue(common_ldap.is_rdn_equal(rdn1, rdn2))

    def test_rdn_multi_ava_diff_order(self):
        rdn1 = ldap.dn.str2dn('cn=cn1+ou=ou1')[0]
        rdn2 = ldap.dn.str2dn('ou=OU1+cn=CN1')[0]
        self.assertTrue(common_ldap.is_rdn_equal(rdn1, rdn2))

    def test_rdn_multi_ava_diff_type(self):
        rdn1 = ldap.dn.str2dn('cn=cn1+ou=ou1')[0]
        rdn2 = ldap.dn.str2dn('cn=cn1+sn=sn1')[0]
        self.assertFalse(common_ldap.is_rdn_equal(rdn1, rdn2))

    def test_rdn_attr_type_case_diff(self):
        rdn1 = ldap.dn.str2dn('cn=cn1')[0]
        rdn2 = ldap.dn.str2dn('CN=cn1')[0]
        self.assertTrue(common_ldap.is_rdn_equal(rdn1, rdn2))

    def test_rdn_attr_type_alias(self):
        rdn1 = ldap.dn.str2dn('cn=cn1')[0]
        rdn2 = ldap.dn.str2dn('2.5.4.3=cn1')[0]
        self.assertFalse(common_ldap.is_rdn_equal(rdn1, rdn2))

    def test_dn_same(self):
        dn = 'cn=Babs Jansen,ou=OpenStack'
        self.assertTrue(common_ldap.is_dn_equal(dn, dn))

    def test_dn_equal_unicode(self):
        dn = u'cn=fäké,ou=OpenStack'
        self.assertTrue(common_ldap.is_dn_equal(dn, dn))

    def test_dn_diff_length(self):
        dn1 = 'cn=Babs Jansen,ou=OpenStack'
        dn2 = 'cn=Babs Jansen,ou=OpenStack,dc=example.com'
        self.assertFalse(common_ldap.is_dn_equal(dn1, dn2))

    def test_dn_equal_rdns(self):
        dn1 = 'cn=Babs Jansen,ou=OpenStack+cn=OpenSource'
        dn2 = 'CN=Babs Jansen,cn=OpenSource+ou=OpenStack'
        self.assertTrue(common_ldap.is_dn_equal(dn1, dn2))

    def test_dn_parsed_dns(self):
        dn_str1 = ldap.dn.str2dn('cn=Babs Jansen,ou=OpenStack+cn=OpenSource')
        dn_str2 = ldap.dn.str2dn('CN=Babs Jansen,cn=OpenSource+ou=OpenStack')
        self.assertTrue(common_ldap.is_dn_equal(dn_str1, dn_str2))

    def test_startswith_under_child(self):
        child = 'cn=Babs Jansen,ou=OpenStack'
        parent = 'ou=OpenStack'
        self.assertTrue(common_ldap.dn_startswith(child, parent))

    def test_startswith_parent(self):
        child = 'cn=Babs Jansen,ou=OpenStack'
        parent = 'ou=OpenStack'
        self.assertFalse(common_ldap.dn_startswith(parent, child))

    def test_startswith_same(self):
        dn = 'cn=Babs Jansen,ou=OpenStack'
        self.assertFalse(common_ldap.dn_startswith(dn, dn))

    def test_startswith_not_parent(self):
        child = 'cn=Babs Jansen,ou=OpenStack'
        parent = 'dc=example.com'
        self.assertFalse(common_ldap.dn_startswith(child, parent))

    def test_startswith_descendant(self):
        descendant = 'cn=Babs Jansen,ou=Keystone,ou=OpenStack,dc=example.com'
        dn = 'ou=OpenStack,dc=example.com'
        self.assertTrue(common_ldap.dn_startswith(descendant, dn))
        descendant = 'uid=12345,ou=Users,dc=example,dc=com'
        dn = 'ou=Users,dc=example,dc=com'
        self.assertTrue(common_ldap.dn_startswith(descendant, dn))

    def test_startswith_parsed_dns(self):
        descendant = ldap.dn.str2dn('cn=Babs Jansen,ou=OpenStack')
        dn = ldap.dn.str2dn('ou=OpenStack')
        self.assertTrue(common_ldap.dn_startswith(descendant, dn))

    def test_startswith_unicode(self):
        child = u'cn=fäké,ou=OpenStäck'
        parent = u'ou=OpenStäck'
        self.assertTrue(common_ldap.dn_startswith(child, parent))