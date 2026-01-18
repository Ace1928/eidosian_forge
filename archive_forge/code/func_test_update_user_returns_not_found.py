import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_update_user_returns_not_found(self):
    user_id = uuid.uuid4().hex
    self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.update_user, user_id, {'id': user_id, 'domain_id': CONF.identity.default_domain_id})