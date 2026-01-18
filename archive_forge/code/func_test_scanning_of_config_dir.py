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
def test_scanning_of_config_dir(self):
    """Test the Manager class scans the config directory.

        The setup for the main tests above load the domain configs directly
        so that the test overrides can be included. This test just makes sure
        that the standard config directory scanning does pick up the relevant
        domain config files.

        """
    self.assertTrue(CONF.identity.domain_specific_drivers_enabled)
    self.load_backends()
    PROVIDERS.identity_api.list_users(domain_scope=self.domains['domain1']['id'])
    self.assertIn('default', PROVIDERS.identity_api.domain_configs)
    self.assertIn(self.domains['domain1']['id'], PROVIDERS.identity_api.domain_configs)
    self.assertIn(self.domains['domain2']['id'], PROVIDERS.identity_api.domain_configs)
    self.assertNotIn(self.domains['domain3']['id'], PROVIDERS.identity_api.domain_configs)
    self.assertNotIn(self.domains['domain4']['id'], PROVIDERS.identity_api.domain_configs)
    conf = PROVIDERS.identity_api.domain_configs.get_domain_conf(self.domains['domain1']['id'])
    self.assertFalse(conf.identity.domain_specific_drivers_enabled)
    self.assertEqual('fake://memory1', conf.ldap.url)