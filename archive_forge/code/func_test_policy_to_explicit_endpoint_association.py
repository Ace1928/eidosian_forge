import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_policy_to_explicit_endpoint_association(self):
    PROVIDERS.endpoint_policy_api.create_policy_association(self.policy[0]['id'], endpoint_id=self.endpoint[0]['id'])
    self._assert_correct_policy(self.endpoint[0], self.policy[0])
    self._assert_correct_endpoints(self.policy[0], [self.endpoint[0]])
    self.assertRaises(exception.NotFound, PROVIDERS.endpoint_policy_api.get_policy_for_endpoint, uuid.uuid4().hex)