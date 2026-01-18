import json
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_list_policies(self):
    policy = unit.new_policy_ref()
    policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
    with self.test_client() as c:
        c.get('/v3/policies', headers=self.headers, expected_status_code=http.client.FORBIDDEN)