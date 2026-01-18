import ldappool
from keystone.common import provider_api
import keystone.conf
from keystone.identity.backends import ldap
from keystone.identity.backends.ldap import common as ldap_common
from keystone.tests import unit
from keystone.tests.unit import fakeldap
from keystone.tests.unit import test_backend_ldap_pool
from keystone.tests.unit import test_ldap_livetest
def test_password_change_with_auth_pool_enabled_long_lifetime(self):
    self.config_fixture.config(group='ldap', auth_pool_connection_lifetime=600)
    old_password = 'my_password'
    new_password = 'new_password'
    user = self._do_password_change_for_one_user(old_password, new_password)
    user.pop('password')
    with self.make_request():
        user_ref = PROVIDERS.identity_api.authenticate(user_id=user['id'], password=old_password)
    self.assertDictEqual(user, user_ref)