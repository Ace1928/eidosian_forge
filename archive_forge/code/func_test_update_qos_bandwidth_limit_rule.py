import copy
from openstack import exceptions
from openstack.network.v2 import qos_bandwidth_limit_rule
from openstack.tests.unit import base
def test_update_qos_bandwidth_limit_rule(self):
    expected_rule = copy.copy(self.mock_rule)
    expected_rule['max_kbps'] = self.rule_max_kbps + 100
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': [self.qos_extension]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_id]), json=self.mock_policy), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_id, 'bandwidth_limit_rules', self.rule_id]), json={'bandwidth_limit_rule': self.mock_rule}), dict(method='PUT', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_id, 'bandwidth_limit_rules', self.rule_id]), json={'bandwidth_limit_rule': expected_rule}, validate=dict(json={'bandwidth_limit_rule': {'max_kbps': self.rule_max_kbps + 100}}))])
    rule = self.cloud.update_qos_bandwidth_limit_rule(self.policy_id, self.rule_id, max_kbps=self.rule_max_kbps + 100)
    self._compare_rules(expected_rule, rule)
    self.assert_calls()