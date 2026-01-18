from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_update_firewall_group_unset_policies(self):
    transformed_params = {'ingress_firewall_policy_id': None, 'egress_firewall_policy_id': None}
    updated_group = deepcopy(self.mock_returned_firewall_group)
    updated_group.update(**transformed_params)
    returned_group = deepcopy(self.mock_returned_firewall_group)
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_groups', self.firewall_group_id), json={'firewall_group': returned_group}), dict(method='PUT', uri=self._make_mock_url('firewall_groups', self.firewall_group_id), json={'firewall_group': updated_group}, validate=dict(json={'firewall_group': transformed_params}))])
    self.assertDictEqual(updated_group, self.cloud.update_firewall_group(self.firewall_group_id, ingress_firewall_policy=None, egress_firewall_policy=None))
    self.assert_calls()