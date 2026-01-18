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
def test_user_id_begins_with_0_and_enabled_bit_mask(self):
    user_id = '0123456'
    bitmask = '225'
    expected_bitmask = 225
    result = [('cn=dummy,dc=example,dc=com', {'user_id': [user_id], 'enabled': [bitmask]})]
    py_result = common_ldap.convert_ldap_result(result)
    self.assertEqual(expected_bitmask, py_result[0][1]['enabled'][0])
    self.assertEqual(user_id, py_result[0][1]['user_id'][0])