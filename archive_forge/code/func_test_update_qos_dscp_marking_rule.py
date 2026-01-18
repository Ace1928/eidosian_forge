import copy
from openstack import exceptions
from openstack.network.v2 import qos_dscp_marking_rule
from openstack.tests.unit import base
def test_update_qos_dscp_marking_rule(self):
    new_dscp_mark_value = 16
    expected_rule = copy.copy(self.mock_rule)
    expected_rule['dscp_mark'] = new_dscp_mark_value
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': self.enabled_neutron_extensions}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_id]), json=self.mock_policy), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_id, 'dscp_marking_rules', self.rule_id]), json={'dscp_marking_rule': self.mock_rule}), dict(method='PUT', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_id, 'dscp_marking_rules', self.rule_id]), json={'dscp_marking_rule': expected_rule}, validate=dict(json={'dscp_marking_rule': {'dscp_mark': new_dscp_mark_value}}))])
    rule = self.cloud.update_qos_dscp_marking_rule(self.policy_id, self.rule_id, dscp_mark=new_dscp_mark_value)
    self._compare_rules(expected_rule, rule)
    self.assert_calls()