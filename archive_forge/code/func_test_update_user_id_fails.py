import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_update_user_id_fails(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user = PROVIDERS.identity_api.create_user(user)
    original_id = user['id']
    user['id'] = 'fake2'
    self.assertRaises(exception.ValidationError, PROVIDERS.identity_api.update_user, original_id, user)
    user_ref = PROVIDERS.identity_api.get_user(original_id)
    self.assertEqual(original_id, user_ref['id'])
    self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.get_user, 'fake2')