from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowSecurityGroupRuleNetwork(TestSecurityGroupRuleNetwork):
    _security_group_rule = network_fakes.FakeSecurityGroupRule.create_one_security_group_rule()
    columns = ('description', 'direction', 'ether_type', 'id', 'port_range_max', 'port_range_min', 'project_id', 'protocol', 'remote_address_group_id', 'remote_group_id', 'remote_ip_prefix', 'security_group_id')
    data = (_security_group_rule.description, _security_group_rule.direction, _security_group_rule.ether_type, _security_group_rule.id, _security_group_rule.port_range_max, _security_group_rule.port_range_min, _security_group_rule.project_id, _security_group_rule.protocol, _security_group_rule.remote_address_group_id, _security_group_rule.remote_group_id, _security_group_rule.remote_ip_prefix, _security_group_rule.security_group_id)

    def setUp(self):
        super(TestShowSecurityGroupRuleNetwork, self).setUp()
        self.network_client.find_security_group_rule = mock.Mock(return_value=self._security_group_rule)
        self.cmd = security_group_rule.ShowSecurityGroupRule(self.app, self.namespace)

    def test_show_no_options(self):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_show_all_options(self):
        arglist = [self._security_group_rule.id]
        verifylist = [('rule', self._security_group_rule.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_security_group_rule.assert_called_once_with(self._security_group_rule.id, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)