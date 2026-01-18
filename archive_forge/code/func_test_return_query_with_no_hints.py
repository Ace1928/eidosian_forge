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
def test_return_query_with_no_hints(self):
    hints = driver_hints.Hints()
    query = uuid.uuid4().hex
    self.assertEqual(query, self.base_ldap.filter_query(hints=hints, query=query))
    self.assertEqual('', self.base_ldap.filter_query(hints=hints))