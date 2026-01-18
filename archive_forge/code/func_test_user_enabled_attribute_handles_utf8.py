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
@mock.patch.object(common_ldap.BaseLdap, '_ldap_get')
def test_user_enabled_attribute_handles_utf8(self, mock_ldap_get):
    self.config_fixture.config(group='ldap', user_enabled_invert=True, user_enabled_attribute='passwordisexpired')
    mock_ldap_get.return_value = (u'uid=123456789,c=us,ou=our_ldap,o=acme.com', {'uid': [123456789], 'mail': [u'shaun@acme.com'], 'passwordisexpired': [u'false'], 'cn': [u'uid=123456789,c=us,ou=our_ldap,o=acme.com']})
    user_api = identity.backends.ldap.UserApi(CONF)
    user_ref = user_api.get('123456789')
    self.assertIs(False, user_ref['enabled'])