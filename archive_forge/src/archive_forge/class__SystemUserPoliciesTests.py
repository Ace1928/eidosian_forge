import json
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _SystemUserPoliciesTests(object):
    """Common default functionality for all system users."""

    def test_user_can_list_policies(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        with self.test_client() as c:
            r = c.get('/v3/policies', headers=self.headers)
            policies = []
            for policy in r.json['policies']:
                policies.append(policy['id'])
            self.assertIn(policy['id'], policies)

    def test_user_can_get_policy(self):
        policy = unit.new_policy_ref()
        policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
        with self.test_client() as c:
            c.get('/v3/policies/%s' % policy['id'], headers=self.headers)