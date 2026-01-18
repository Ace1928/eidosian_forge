from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_update_firewall_group(self):
    params = {'description': 'updated!', 'egress_firewall_policy': self.mock_egress_policy['name'], 'ingress_firewall_policy': self.mock_ingress_policy['name'], 'ports': [self.mock_port['name']]}
    updated_group = deepcopy(self.mock_returned_firewall_group)
    updated_group['description'] = params['description']
    returned_group = deepcopy(self.mock_returned_firewall_group)
    returned_group.update(ingress_firewall_policy_id=None, egress_firewall_policy_id=None, ports=[])
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_groups', self.firewall_group_name), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_groups', name=self.firewall_group_name), json={'firewall_groups': [returned_group]}), dict(method='GET', uri=self._make_mock_url('firewall_policies', self.mock_egress_policy['name']), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_policies', name=self.mock_egress_policy['name']), json={'firewall_policies': [deepcopy(self.mock_egress_policy)]}), dict(method='GET', uri=self._make_mock_url('firewall_policies', self.mock_ingress_policy['name']), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_policies', name=self.mock_ingress_policy['name']), json={'firewall_policies': [deepcopy(self.mock_ingress_policy)]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports', self.mock_port['name']]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports'], qs_elements=['name=%s' % self.mock_port['name']]), json={'ports': [self.mock_port]}), dict(method='PUT', uri=self._make_mock_url('firewall_groups', self.firewall_group_id), json={'firewall_group': updated_group}, validate=dict(json={'firewall_group': {'description': params['description'], 'egress_firewall_policy_id': self.mock_egress_policy['id'], 'ingress_firewall_policy_id': self.mock_ingress_policy['id'], 'ports': [self.mock_port['id']]}}))])
    self.assertDictEqual(updated_group, self.cloud.update_firewall_group(self.firewall_group_name, **params))
    self.assert_calls()