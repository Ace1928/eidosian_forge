import copy
from openstack import exceptions
from openstack.network.v2 import qos_policy as _policy
from openstack.tests.unit import base
def test_update_qos_policy_no_qos_default_extension(self):
    expected_policy = copy.copy(self.mock_policy)
    expected_policy['name'] = 'goofy'
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': [self.qos_extension]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': [self.qos_extension]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_id]), json=self.mock_policy), dict(method='PUT', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_id]), json={'policy': expected_policy}, validate=dict(json={'policy': {'name': 'goofy'}}))])
    policy = self.cloud.update_qos_policy(self.policy_id, name='goofy', default=True)
    self._compare_policies(expected_policy, policy)
    self.assert_calls()