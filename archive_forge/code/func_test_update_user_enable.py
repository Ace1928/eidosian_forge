import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_update_user_enable(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user = PROVIDERS.identity_api.create_user(user)
    user_ref = PROVIDERS.identity_api.get_user(user['id'])
    self.assertTrue(user_ref['enabled'])
    user['enabled'] = False
    PROVIDERS.identity_api.update_user(user['id'], user)
    user_ref = PROVIDERS.identity_api.get_user(user['id'])
    self.assertEqual(user['enabled'], user_ref['enabled'])
    del user['enabled']
    PROVIDERS.identity_api.update_user(user['id'], user)
    user_ref = PROVIDERS.identity_api.get_user(user['id'])
    self.assertFalse(user_ref['enabled'])
    user['enabled'] = True
    PROVIDERS.identity_api.update_user(user['id'], user)
    user_ref = PROVIDERS.identity_api.get_user(user['id'])
    self.assertEqual(user['enabled'], user_ref['enabled'])
    del user['enabled']
    PROVIDERS.identity_api.update_user(user['id'], user)
    user_ref = PROVIDERS.identity_api.get_user(user['id'])
    self.assertTrue(user_ref['enabled'])