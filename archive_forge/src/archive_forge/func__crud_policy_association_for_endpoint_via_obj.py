import uuid
from keystoneclient.tests.unit.v3 import test_endpoint_filter
from keystoneclient.tests.unit.v3 import utils
def _crud_policy_association_for_endpoint_via_obj(self, http_action, manager_action):
    policy_ref = self.new_policy_ref()
    endpoint_ref = self.new_endpoint_ref()
    policy = self.client.policies.resource_class(self.client.policies, policy_ref, loaded=True)
    endpoint = self.client.endpoints.resource_class(self.client.endpoints, endpoint_ref, loaded=True)
    self.stub_url(http_action, ['policies', policy_ref['id'], self.manager.OS_EP_POLICY_EXT, 'endpoints', endpoint_ref['id']], status_code=204)
    manager_action(policy=policy, endpoint=endpoint)