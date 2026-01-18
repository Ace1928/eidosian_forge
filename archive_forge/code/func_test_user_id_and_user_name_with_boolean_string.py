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
def test_user_id_and_user_name_with_boolean_string(self):
    boolean_strings = ['TRUE', 'FALSE', 'true', 'false', 'True', 'False', 'TrUeFaLse']
    for user_name in boolean_strings:
        user_id = uuid.uuid4().hex
        result = [('cn=dummy,dc=example,dc=com', {'user_id': [user_id], 'user_name': [user_name]})]
        py_result = common_ldap.convert_ldap_result(result)
        self.assertEqual(user_name, py_result[0][1]['user_name'][0])