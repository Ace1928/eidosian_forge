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
def test_user_with_missing_id(self):
    ldap_ = PROVIDERS.identity_api.driver.user.get_connection()
    ldap_id_field = 'sn'
    ldap_id_value = uuid.uuid4().hex
    dn = '%s=%s,ou=Users,cn=example,cn=com' % (ldap_id_field, ldap_id_value)
    modlist = [('objectClass', ['person', 'inetOrgPerson']), (ldap_id_field, [ldap_id_value]), ('mail', ['email@example.com']), ('userPassword', [uuid.uuid4().hex])]
    ldap_.add_s(dn, modlist)
    users = PROVIDERS.identity_api.driver.user.get_all()
    self.assertThat(users, matchers.HasLength(len(default_fixtures.USERS)))