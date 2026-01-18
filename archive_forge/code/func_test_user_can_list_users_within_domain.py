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
def test_user_can_list_users_within_domain(self):
    user = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=self.domain_id))
    with self.test_client() as c:
        r = c.get('/v3/users', headers=self.headers)
        self.assertEqual(2, len(r.json['users']))
        user_ids = []
        for user in r.json['users']:
            user_ids.append(user['id'])
        self.assertIn(self.user_id, user_ids)
        self.assertIn(user['id'], user_ids)