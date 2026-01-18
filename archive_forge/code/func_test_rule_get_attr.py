from unittest import mock
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_rule_get_attr(self):
    self.minimum_packet_rate_rule.resource_id = 'test rule'
    rule = {'minimum_packet_rate_rule': {'id': 'cf0eab12-ef8b-4a62-98d0-70576583c17a', 'min_kpps': 1000, 'direction': 'egress', 'tenant_id': 'd66c74c01d6c41b9846088c1ad9634d0'}}
    self.neutronclient.show_minimum_packet_rate_rule.return_value = rule
    self.assertEqual(rule['minimum_packet_rate_rule'], self.minimum_packet_rate_rule.FnGetAtt('show'))
    self.neutronclient.show_minimum_packet_rate_rule.assert_called_once_with(self.minimum_packet_rate_rule.resource_id, self.policy_id)