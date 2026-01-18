import uuid
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_update_policy_returns_not_found(self):
    ref = unit.new_policy_ref()
    self.assertRaises(exception.PolicyNotFound, PROVIDERS.policy_api.update_policy, ref['id'], ref)