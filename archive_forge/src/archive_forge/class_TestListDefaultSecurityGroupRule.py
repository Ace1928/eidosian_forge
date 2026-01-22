from unittest import mock
from unittest.mock import call
import uuid
from openstack.network.v2 import _proxy
from openstack.network.v2 import (
from openstack.test import fakes as sdk_fakes
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import default_security_group_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListDefaultSecurityGroupRule(TestDefaultSecurityGroupRule):
    _default_sg_rule_tcp = sdk_fakes.generate_fake_resource(_default_security_group_rule.DefaultSecurityGroupRule, **{'protocol': 'tcp', 'port_range_max': 80, 'port_range_min': 80})
    _default_sg_rule_icmp = sdk_fakes.generate_fake_resource(_default_security_group_rule.DefaultSecurityGroupRule, **{'protocol': 'icmp', 'remote_ip_prefix': '10.0.2.0/24'})
    _default_sg_rules = [_default_sg_rule_tcp, _default_sg_rule_icmp]
    expected_columns = ('ID', 'IP Protocol', 'Ethertype', 'IP Range', 'Port Range', 'Direction', 'Remote Security Group', 'Remote Address Group', 'Used in default Security Group', 'Used in custom Security Group')
    expected_data = []
    expected_data_no_group = []
    for _default_sg_rule in _default_sg_rules:
        expected_data.append((_default_sg_rule.id, _default_sg_rule.protocol, _default_sg_rule.ether_type, _default_sg_rule.remote_ip_prefix, network_utils.format_network_port_range(_default_sg_rule), _default_sg_rule.direction, _default_sg_rule.remote_group_id, _default_sg_rule.remote_address_group_id, _default_sg_rule.used_in_default_sg, _default_sg_rule.used_in_non_default_sg))

    def setUp(self):
        super(TestListDefaultSecurityGroupRule, self).setUp()
        self.sdk_client.default_security_group_rules.return_value = self._default_sg_rules
        self.cmd = default_security_group_rule.ListDefaultSecurityGroupRule(self.app, self.namespace)

    def test_list_default(self):
        self._default_sg_rule_tcp.port_range_min = 80
        parsed_args = self.check_parser(self.cmd, [], [])
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.default_security_group_rules.assert_called_once_with(**{})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, list(data))

    def test_list_with_protocol(self):
        self._default_sg_rule_tcp.port_range_min = 80
        arglist = ['--protocol', 'tcp']
        verifylist = [('protocol', 'tcp')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.default_security_group_rules.assert_called_once_with(**{'protocol': 'tcp'})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, list(data))

    def test_list_with_ingress(self):
        self._default_sg_rule_tcp.port_range_min = 80
        arglist = ['--ingress']
        verifylist = [('ingress', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.default_security_group_rules.assert_called_once_with(**{'direction': 'ingress'})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, list(data))

    def test_list_with_wrong_egress(self):
        self._default_sg_rule_tcp.port_range_min = 80
        arglist = ['--egress']
        verifylist = [('egress', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.default_security_group_rules.assert_called_once_with(**{'direction': 'egress'})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, list(data))