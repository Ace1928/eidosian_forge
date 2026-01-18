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
@mock.patch.object(fakeldap.FakeLdap, 'search_s')
def test_search_s_sizelimit_exceeded(self, mock_search_s):
    mock_search_s.side_effect = ldap.SIZELIMIT_EXCEEDED
    conn = PROVIDERS.identity_api.user.get_connection()
    self.assertRaises(ks_exception.LDAPSizeLimitExceeded, conn.search_s, 'dc=example,dc=test', ldap.SCOPE_SUBTREE)