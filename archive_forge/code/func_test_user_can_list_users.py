import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import user as up
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_can_list_users(self):
    expected_user_ids = []
    for _ in range(3):
        user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=CONF.identity.default_domain_id))
        expected_user_ids.append(user['id'])
    with self.test_client() as c:
        r = c.get('/v3/users', headers=self.headers)
        returned_user_ids = []
        for user in r.json['users']:
            returned_user_ids.append(user['id'])
        for user_id in expected_user_ids:
            self.assertIn(user_id, returned_user_ids)