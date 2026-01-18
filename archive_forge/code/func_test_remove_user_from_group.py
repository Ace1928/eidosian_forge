from oslo_config import fixture as config_fixture
from keystone.identity.backends import ldap
from keystone.tests.unit import core
from keystone.tests.unit.identity.backends import test_base
from keystone.tests.unit.ksfixtures import ldapdb
def test_remove_user_from_group(self):
    self.skip_test_overrides('N/A: LDAP has no write support')