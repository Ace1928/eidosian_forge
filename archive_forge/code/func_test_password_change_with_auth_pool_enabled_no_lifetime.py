import ldappool
from keystone.common import provider_api
import keystone.conf
from keystone.identity.backends import ldap
from keystone.identity.backends.ldap import common as ldap_common
from keystone.tests import unit
from keystone.tests.unit import fakeldap
from keystone.tests.unit import test_backend_ldap_pool
from keystone.tests.unit import test_ldap_livetest
def test_password_change_with_auth_pool_enabled_no_lifetime(self):
    self.config_fixture.config(group='ldap', auth_pool_connection_lifetime=0)
    old_password = 'my_password'
    new_password = 'new_password'
    user = self._do_password_change_for_one_user(old_password, new_password)
    self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, context={}, user_id=user['id'], password=old_password)