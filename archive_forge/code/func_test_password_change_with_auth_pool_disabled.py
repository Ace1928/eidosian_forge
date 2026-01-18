import ldappool
from keystone.common import provider_api
import keystone.conf
from keystone.identity.backends import ldap
from keystone.identity.backends.ldap import common as ldap_common
from keystone.tests import unit
from keystone.tests.unit import fakeldap
from keystone.tests.unit import test_backend_ldap_pool
from keystone.tests.unit import test_ldap_livetest
def test_password_change_with_auth_pool_disabled(self):
    self.config_fixture.config(group='ldap', use_auth_pool=False)
    old_password = self.user_sna['password']
    self.test_password_change_with_pool()
    self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, context={}, user_id=self.user_sna['id'], password=old_password)