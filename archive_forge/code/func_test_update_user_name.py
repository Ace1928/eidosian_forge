import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_update_user_name(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user = PROVIDERS.identity_api.create_user(user)
    user_ref = PROVIDERS.identity_api.get_user(user['id'])
    self.assertEqual(user['name'], user_ref['name'])
    changed_name = user_ref['name'] + '_changed'
    user_ref['name'] = changed_name
    updated_user = PROVIDERS.identity_api.update_user(user_ref['id'], user_ref)
    updated_user.pop('extra', None)
    self.assertDictEqual(user_ref, updated_user)
    user_ref = PROVIDERS.identity_api.get_user(user_ref['id'])
    self.assertEqual(changed_name, user_ref['name'])