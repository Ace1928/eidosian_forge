import json
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_update_policy(self):
    policy = unit.new_policy_ref()
    policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
    update = {'policy': {'name': uuid.uuid4().hex}}
    with self.test_client() as c:
        c.patch('/v3/policies/%s' % policy['id'], json=update, headers=self.headers)