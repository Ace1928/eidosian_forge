from unittest import mock
import fixtures
import ldappool
from keystone.common import provider_api
import keystone.conf
from keystone.identity.backends import ldap
from keystone.identity.backends.ldap import common as common_ldap
from keystone.tests import unit
from keystone.tests.unit import fakeldap
from keystone.tests.unit import test_backend_ldap
def test_password_change_with_pool(self):
    old_password = self.user_sna['password']
    self.cleanup_pools()
    with self.make_request():
        user_ref = PROVIDERS.identity_api.authenticate(user_id=self.user_sna['id'], password=self.user_sna['password'])
    self.user_sna.pop('password')
    self.user_sna['enabled'] = True
    self.assertUserDictEqual(self.user_sna, user_ref)
    new_password = 'new_password'
    user_ref['password'] = new_password
    PROVIDERS.identity_api.update_user(user_ref['id'], user_ref)
    with self.make_request():
        user_ref2 = PROVIDERS.identity_api.authenticate(user_id=self.user_sna['id'], password=new_password)
    user_ref.pop('password')
    self.assertUserDictEqual(user_ref, user_ref2)
    with self.make_request():
        self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=self.user_sna['id'], password=old_password)