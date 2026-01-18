from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_remove_rule_from_policy_not_associated(self):
    rule = FirewallRule(**TestFirewallRule._mock_firewall_rule_attrs).to_dict()
    policy = deepcopy(self.mock_firewall_policy)
    del policy['firewall_rules'][0]
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_policies', policy['id']), json={'firewall_policy': policy}), dict(method='GET', uri=self._make_mock_url('firewall_rules', rule['id']), json={'firewall_rule': rule})])
    with mock.patch.object(self.cloud.network, 'remove_rule_from_policy'), mock.patch.object(self.cloud.log, 'debug'):
        r = self.cloud.remove_rule_from_policy(policy['id'], rule['id'])
        self.assertDictEqual(policy, r.to_dict())
        self.assert_calls()
        self.cloud.log.debug.assert_called_once()
        self.cloud.network.remove_rule_from_policy.assert_not_called()