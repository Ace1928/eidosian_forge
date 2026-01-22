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
class TestCreateDefaultSecurityGroupRule(TestDefaultSecurityGroupRule):
    expected_columns = ('description', 'direction', 'ether_type', 'id', 'port_range_max', 'port_range_min', 'protocol', 'remote_address_group_id', 'remote_group_id', 'remote_ip_prefix', 'used_in_default_sg', 'used_in_non_default_sg')
    expected_data = None

    def _setup_default_security_group_rule(self, attrs=None):
        default_security_group_rule_attrs = {'description': 'default-security-group-rule-description-' + uuid.uuid4().hex, 'direction': 'ingress', 'ether_type': 'IPv4', 'id': 'default-security-group-rule-id-' + uuid.uuid4().hex, 'port_range_max': None, 'port_range_min': None, 'protocol': None, 'remote_group_id': None, 'remote_address_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'location': 'MUNCHMUNCHMUNCH', 'used_in_default_sg': False, 'used_in_non_default_sg': True}
        attrs = attrs or {}
        default_security_group_rule_attrs.update(attrs)
        self._default_sg_rule = sdk_fakes.generate_fake_resource(_default_security_group_rule.DefaultSecurityGroupRule, **default_security_group_rule_attrs)
        self.sdk_client.create_default_security_group_rule.return_value = self._default_sg_rule
        self.expected_data = (self._default_sg_rule.description, self._default_sg_rule.direction, self._default_sg_rule.ether_type, self._default_sg_rule.id, self._default_sg_rule.port_range_max, self._default_sg_rule.port_range_min, self._default_sg_rule.protocol, self._default_sg_rule.remote_address_group_id, self._default_sg_rule.remote_group_id, self._default_sg_rule.remote_ip_prefix, self._default_sg_rule.used_in_default_sg, self._default_sg_rule.used_in_non_default_sg)

    def setUp(self):
        super(TestCreateDefaultSecurityGroupRule, self).setUp()
        self.cmd = default_security_group_rule.CreateDefaultSecurityGroupRule(self.app, self.namespace)

    def test_create_all_remote_options(self):
        arglist = ['--remote-ip', '10.10.0.0/24', '--remote-group', 'test-remote-group-id', '--remote-address-group', 'test-remote-address-group-id']
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_create_bad_ethertype(self):
        arglist = ['--ethertype', 'foo']
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_lowercase_ethertype(self):
        arglist = ['--ethertype', 'ipv4']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertEqual('IPv4', parsed_args.ethertype)

    def test_lowercase_v6_ethertype(self):
        arglist = ['--ethertype', 'ipv6']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertEqual('IPv6', parsed_args.ethertype)

    def test_proper_case_ethertype(self):
        arglist = ['--ethertype', 'IPv6']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertEqual('IPv6', parsed_args.ethertype)

    def test_create_all_port_range_options(self):
        arglist = ['--dst-port', '80:80', '--icmp-type', '3', '--icmp-code', '1']
        verifylist = [('dst_port', (80, 80)), ('icmp_type', 3), ('icmp_code', 1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_default_rule(self):
        self._setup_default_security_group_rule({'protocol': 'tcp', 'port_range_max': 443, 'port_range_min': 443})
        arglist = ['--protocol', 'tcp', '--dst-port', str(self._default_sg_rule.port_range_min)]
        verifylist = [('dst_port', (self._default_sg_rule.port_range_min, self._default_sg_rule.port_range_max))]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.create_default_security_group_rule.assert_called_once_with(**{'direction': self._default_sg_rule.direction, 'ethertype': self._default_sg_rule.ether_type, 'port_range_max': self._default_sg_rule.port_range_max, 'port_range_min': self._default_sg_rule.port_range_min, 'protocol': self._default_sg_rule.protocol, 'remote_ip_prefix': self._default_sg_rule.remote_ip_prefix, 'used_in_default_sg': False, 'used_in_non_default_sg': True})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_protocol_any(self):
        self._setup_default_security_group_rule({'protocol': None, 'remote_ip_prefix': '10.0.2.0/24'})
        arglist = ['--protocol', 'any', '--remote-ip', self._default_sg_rule.remote_ip_prefix]
        verifylist = [('protocol', 'any'), ('remote_ip', self._default_sg_rule.remote_ip_prefix)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.create_default_security_group_rule.assert_called_once_with(**{'direction': self._default_sg_rule.direction, 'ethertype': self._default_sg_rule.ether_type, 'protocol': self._default_sg_rule.protocol, 'remote_ip_prefix': self._default_sg_rule.remote_ip_prefix, 'used_in_default_sg': False, 'used_in_non_default_sg': True})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_remote_address_group(self):
        self._setup_default_security_group_rule({'protocol': 'icmp', 'remote_address_group_id': 'remote-address-group-id'})
        arglist = ['--protocol', 'icmp', '--remote-address-group', self._default_sg_rule.remote_address_group_id]
        verifylist = [('remote_address_group', self._default_sg_rule.remote_address_group_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.create_default_security_group_rule.assert_called_once_with(**{'direction': self._default_sg_rule.direction, 'ethertype': self._default_sg_rule.ether_type, 'protocol': self._default_sg_rule.protocol, 'remote_address_group_id': self._default_sg_rule.remote_address_group_id, 'used_in_default_sg': False, 'used_in_non_default_sg': True})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_remote_group(self):
        self._setup_default_security_group_rule({'protocol': 'tcp', 'port_range_max': 22, 'port_range_min': 22})
        arglist = ['--protocol', 'tcp', '--dst-port', str(self._default_sg_rule.port_range_min), '--ingress', '--remote-group', 'remote-group-id']
        verifylist = [('dst_port', (self._default_sg_rule.port_range_min, self._default_sg_rule.port_range_max)), ('ingress', True), ('remote_group', 'remote-group-id')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.create_default_security_group_rule.assert_called_once_with(**{'direction': self._default_sg_rule.direction, 'ethertype': self._default_sg_rule.ether_type, 'port_range_max': self._default_sg_rule.port_range_max, 'port_range_min': self._default_sg_rule.port_range_min, 'protocol': self._default_sg_rule.protocol, 'remote_group_id': 'remote-group-id', 'used_in_default_sg': False, 'used_in_non_default_sg': True})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_source_group(self):
        self._setup_default_security_group_rule({'remote_group_id': 'remote-group-id'})
        arglist = ['--ingress', '--remote-group', 'remote-group-id']
        verifylist = [('ingress', True), ('remote_group', 'remote-group-id')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.create_default_security_group_rule.assert_called_once_with(**{'direction': self._default_sg_rule.direction, 'ethertype': self._default_sg_rule.ether_type, 'protocol': self._default_sg_rule.protocol, 'remote_group_id': 'remote-group-id', 'used_in_default_sg': False, 'used_in_non_default_sg': True})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_source_ip(self):
        self._setup_default_security_group_rule({'protocol': 'icmp', 'remote_ip_prefix': '10.0.2.0/24'})
        arglist = ['--protocol', self._default_sg_rule.protocol, '--remote-ip', self._default_sg_rule.remote_ip_prefix]
        verifylist = [('protocol', self._default_sg_rule.protocol), ('remote_ip', self._default_sg_rule.remote_ip_prefix)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.create_default_security_group_rule.assert_called_once_with(**{'direction': self._default_sg_rule.direction, 'ethertype': self._default_sg_rule.ether_type, 'protocol': self._default_sg_rule.protocol, 'remote_ip_prefix': self._default_sg_rule.remote_ip_prefix, 'used_in_default_sg': False, 'used_in_non_default_sg': True})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_remote_ip(self):
        self._setup_default_security_group_rule({'protocol': 'icmp', 'remote_ip_prefix': '10.0.2.0/24'})
        arglist = ['--protocol', self._default_sg_rule.protocol, '--remote-ip', self._default_sg_rule.remote_ip_prefix]
        verifylist = [('protocol', self._default_sg_rule.protocol), ('remote_ip', self._default_sg_rule.remote_ip_prefix)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.create_default_security_group_rule.assert_called_once_with(**{'direction': self._default_sg_rule.direction, 'ethertype': self._default_sg_rule.ether_type, 'protocol': self._default_sg_rule.protocol, 'remote_ip_prefix': self._default_sg_rule.remote_ip_prefix, 'used_in_default_sg': False, 'used_in_non_default_sg': True})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_tcp_with_icmp_type(self):
        arglist = ['--protocol', 'tcp', '--icmp-type', '15']
        verifylist = [('protocol', 'tcp'), ('icmp_type', 15)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_icmp_code(self):
        arglist = ['--protocol', '1', '--icmp-code', '1']
        verifylist = [('protocol', '1'), ('icmp_code', 1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_icmp_code_zero(self):
        self._setup_default_security_group_rule({'port_range_min': 15, 'port_range_max': 0, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
        arglist = ['--protocol', self._default_sg_rule.protocol, '--icmp-type', str(self._default_sg_rule.port_range_min), '--icmp-code', str(self._default_sg_rule.port_range_max)]
        verifylist = [('protocol', self._default_sg_rule.protocol), ('icmp_code', self._default_sg_rule.port_range_max), ('icmp_type', self._default_sg_rule.port_range_min)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_icmp_code_greater_than_zero(self):
        self._setup_default_security_group_rule({'port_range_min': 15, 'port_range_max': 18, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
        arglist = ['--protocol', self._default_sg_rule.protocol, '--icmp-type', str(self._default_sg_rule.port_range_min), '--icmp-code', str(self._default_sg_rule.port_range_max)]
        verifylist = [('protocol', self._default_sg_rule.protocol), ('icmp_type', self._default_sg_rule.port_range_min), ('icmp_code', self._default_sg_rule.port_range_max)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_icmp_code_negative_value(self):
        self._setup_default_security_group_rule({'port_range_min': 15, 'port_range_max': None, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
        arglist = ['--protocol', self._default_sg_rule.protocol, '--icmp-type', str(self._default_sg_rule.port_range_min), '--icmp-code', '-2']
        verifylist = [('protocol', self._default_sg_rule.protocol), ('icmp_type', self._default_sg_rule.port_range_min), ('icmp_code', -2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_icmp_type(self):
        self._setup_default_security_group_rule({'port_range_min': 15, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
        arglist = ['--icmp-type', str(self._default_sg_rule.port_range_min), '--protocol', self._default_sg_rule.protocol]
        verifylist = [('dst_port', None), ('icmp_type', self._default_sg_rule.port_range_min), ('icmp_code', None), ('protocol', self._default_sg_rule.protocol)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.create_default_security_group_rule.assert_called_once_with(**{'direction': self._default_sg_rule.direction, 'ethertype': self._default_sg_rule.ether_type, 'port_range_min': self._default_sg_rule.port_range_min, 'protocol': self._default_sg_rule.protocol, 'remote_ip_prefix': self._default_sg_rule.remote_ip_prefix, 'used_in_default_sg': False, 'used_in_non_default_sg': True})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_icmp_type_zero(self):
        self._setup_default_security_group_rule({'port_range_min': 0, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
        arglist = ['--icmp-type', str(self._default_sg_rule.port_range_min), '--protocol', self._default_sg_rule.protocol]
        verifylist = [('dst_port', None), ('icmp_type', self._default_sg_rule.port_range_min), ('icmp_code', None), ('protocol', self._default_sg_rule.protocol)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.create_default_security_group_rule.assert_called_once_with(**{'direction': self._default_sg_rule.direction, 'ethertype': self._default_sg_rule.ether_type, 'port_range_min': self._default_sg_rule.port_range_min, 'protocol': self._default_sg_rule.protocol, 'remote_ip_prefix': self._default_sg_rule.remote_ip_prefix, 'used_in_default_sg': False, 'used_in_non_default_sg': True})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_icmp_type_greater_than_zero(self):
        self._setup_default_security_group_rule({'port_range_min': 13, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
        arglist = ['--icmp-type', str(self._default_sg_rule.port_range_min), '--protocol', self._default_sg_rule.protocol]
        verifylist = [('dst_port', None), ('icmp_type', self._default_sg_rule.port_range_min), ('icmp_code', None), ('protocol', self._default_sg_rule.protocol)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.create_default_security_group_rule.assert_called_once_with(**{'direction': self._default_sg_rule.direction, 'ethertype': self._default_sg_rule.ether_type, 'port_range_min': self._default_sg_rule.port_range_min, 'protocol': self._default_sg_rule.protocol, 'remote_ip_prefix': self._default_sg_rule.remote_ip_prefix, 'used_in_default_sg': False, 'used_in_non_default_sg': True})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_icmp_type_negative_value(self):
        self._setup_default_security_group_rule({'port_range_min': None, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
        arglist = ['--icmp-type', '-13', '--protocol', self._default_sg_rule.protocol]
        verifylist = [('dst_port', None), ('icmp_type', -13), ('icmp_code', None), ('protocol', self._default_sg_rule.protocol)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.create_default_security_group_rule.assert_called_once_with(**{'direction': self._default_sg_rule.direction, 'ethertype': self._default_sg_rule.ether_type, 'protocol': self._default_sg_rule.protocol, 'remote_ip_prefix': self._default_sg_rule.remote_ip_prefix, 'used_in_default_sg': False, 'used_in_non_default_sg': True})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_ipv6_icmp_type_code(self):
        self._setup_default_security_group_rule({'ether_type': 'IPv6', 'port_range_min': 139, 'port_range_max': 2, 'protocol': 'ipv6-icmp', 'remote_ip_prefix': '::/0'})
        arglist = ['--icmp-type', str(self._default_sg_rule.port_range_min), '--icmp-code', str(self._default_sg_rule.port_range_max), '--protocol', self._default_sg_rule.protocol]
        verifylist = [('dst_port', None), ('icmp_type', self._default_sg_rule.port_range_min), ('icmp_code', self._default_sg_rule.port_range_max), ('protocol', self._default_sg_rule.protocol)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.create_default_security_group_rule.assert_called_once_with(**{'direction': self._default_sg_rule.direction, 'ethertype': self._default_sg_rule.ether_type, 'port_range_min': self._default_sg_rule.port_range_min, 'port_range_max': self._default_sg_rule.port_range_max, 'protocol': self._default_sg_rule.protocol, 'remote_ip_prefix': self._default_sg_rule.remote_ip_prefix, 'used_in_default_sg': False, 'used_in_non_default_sg': True})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_icmpv6_type(self):
        self._setup_default_security_group_rule({'ether_type': 'IPv6', 'port_range_min': 139, 'protocol': 'icmpv6', 'remote_ip_prefix': '::/0'})
        arglist = ['--icmp-type', str(self._default_sg_rule.port_range_min), '--protocol', self._default_sg_rule.protocol]
        verifylist = [('dst_port', None), ('icmp_type', self._default_sg_rule.port_range_min), ('icmp_code', None), ('protocol', self._default_sg_rule.protocol)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.create_default_security_group_rule.assert_called_once_with(**{'direction': self._default_sg_rule.direction, 'ethertype': self._default_sg_rule.ether_type, 'port_range_min': self._default_sg_rule.port_range_min, 'protocol': self._default_sg_rule.protocol, 'remote_ip_prefix': self._default_sg_rule.remote_ip_prefix, 'used_in_default_sg': False, 'used_in_non_default_sg': True})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_with_description(self):
        self._setup_default_security_group_rule({'description': 'Setting SGR'})
        arglist = ['--description', self._default_sg_rule.description]
        verifylist = [('description', self._default_sg_rule.description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.sdk_client.create_default_security_group_rule.assert_called_once_with(**{'description': self._default_sg_rule.description, 'direction': self._default_sg_rule.direction, 'ethertype': self._default_sg_rule.ether_type, 'protocol': self._default_sg_rule.protocol, 'remote_ip_prefix': self._default_sg_rule.remote_ip_prefix, 'used_in_default_sg': False, 'used_in_non_default_sg': True})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)