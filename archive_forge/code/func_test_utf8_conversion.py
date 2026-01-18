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
def test_utf8_conversion(self):
    value_unicode = u'fäké1'
    value_utf8 = value_unicode.encode('utf-8')
    result_utf8 = common_ldap.utf8_encode(value_unicode)
    self.assertEqual(value_utf8, result_utf8)
    result_utf8 = common_ldap.utf8_encode(value_utf8)
    self.assertEqual(value_utf8, result_utf8)
    result_unicode = common_ldap.utf8_decode(value_utf8)
    self.assertEqual(value_unicode, result_unicode)
    result_unicode = common_ldap.utf8_decode(value_unicode)
    self.assertEqual(value_unicode, result_unicode)
    self.assertRaises(TypeError, common_ldap.utf8_encode, 100)
    result_unicode = common_ldap.utf8_decode(100)
    self.assertEqual(u'100', result_unicode)