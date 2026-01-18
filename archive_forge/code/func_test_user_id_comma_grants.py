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
def test_user_id_comma_grants(self):
    """List user and group grants, even with a comma in the user's ID."""
    user_id = u'Doe, John'
    user = self.new_user_ref(id=user_id, domain_id=CONF.identity.default_domain_id)
    PROVIDERS.identity_api.driver.create_user(user_id, user)
    ref_list = PROVIDERS.identity_api.list_users()
    public_user_id = None
    for ref in ref_list:
        if ref['name'] == user['name']:
            public_user_id = ref['id']
            break
    role_id = default_fixtures.MEMBER_ROLE_ID
    project_id = self.project_baz['id']
    PROVIDERS.assignment_api.create_grant(role_id, user_id=public_user_id, project_id=project_id)
    role_ref = PROVIDERS.assignment_api.get_grant(role_id, user_id=public_user_id, project_id=project_id)
    self.assertEqual(role_id, role_ref['id'])