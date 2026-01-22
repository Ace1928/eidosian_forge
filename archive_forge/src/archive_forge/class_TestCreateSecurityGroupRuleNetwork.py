from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateSecurityGroupRuleNetwork(TestSecurityGroupRuleNetwork):
    project = identity_fakes.FakeProject.create_one_project()
    domain = identity_fakes.FakeDomain.create_one_domain()
    _security_group_rule = None
    _security_group = network_fakes.FakeSecurityGroup.create_one_security_group()
    _address_group = network_fakes.create_one_address_group()
    expected_columns = ('description', 'direction', 'ether_type', 'id', 'port_range_max', 'port_range_min', 'project_id', 'protocol', 'remote_address_group_id', 'remote_group_id', 'remote_ip_prefix', 'security_group_id')
    expected_data = None

    def _setup_security_group_rule(self, attrs=None):
        self._security_group_rule = network_fakes.FakeSecurityGroupRule.create_one_security_group_rule(attrs)
        self.network_client.create_security_group_rule = mock.Mock(return_value=self._security_group_rule)
        self.expected_data = (self._security_group_rule.description, self._security_group_rule.direction, self._security_group_rule.ether_type, self._security_group_rule.id, self._security_group_rule.port_range_max, self._security_group_rule.port_range_min, self._security_group_rule.project_id, self._security_group_rule.protocol, self._security_group_rule.remote_address_group_id, self._security_group_rule.remote_group_id, self._security_group_rule.remote_ip_prefix, self._security_group_rule.security_group_id)

    def setUp(self):
        super(TestCreateSecurityGroupRuleNetwork, self).setUp()
        self.network_client.find_security_group = mock.Mock(return_value=self._security_group)
        self.network_client.find_address_group = mock.Mock(return_value=self._address_group)
        self.projects_mock.get.return_value = self.project
        self.domains_mock.get.return_value = self.domain
        self.cmd = security_group_rule.CreateSecurityGroupRule(self.app, self.namespace)

    def test_create_no_options(self):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_create_all_remote_options(self):
        arglist = ['--remote-ip', '10.10.0.0/24', '--remote-group', self._security_group.id, '--remote-address-group', self._address_group.id, self._security_group.id]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_create_bad_ethertype(self):
        arglist = ['--ethertype', 'foo', self._security_group.id]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_lowercase_ethertype(self):
        arglist = ['--ethertype', 'ipv4', self._security_group.id]
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertEqual('IPv4', parsed_args.ethertype)

    def test_lowercase_v6_ethertype(self):
        arglist = ['--ethertype', 'ipv6', self._security_group.id]
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertEqual('IPv6', parsed_args.ethertype)

    def test_proper_case_ethertype(self):
        arglist = ['--ethertype', 'IPv6', self._security_group.id]
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertEqual('IPv6', parsed_args.ethertype)

    def test_create_all_protocol_options(self):
        arglist = ['--protocol', 'tcp', '--proto', 'tcp', self._security_group.id]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_create_all_port_range_options(self):
        arglist = ['--dst-port', '80:80', '--icmp-type', '3', '--icmp-code', '1', self._security_group.id]
        verifylist = [('dst_port', (80, 80)), ('icmp_type', 3), ('icmp_code', 1), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_default_rule(self):
        self._setup_security_group_rule({'protocol': 'tcp', 'port_range_max': 443, 'port_range_min': 443})
        arglist = ['--protocol', 'tcp', '--dst-port', str(self._security_group_rule.port_range_min), self._security_group.id]
        verifylist = [('dst_port', (self._security_group_rule.port_range_min, self._security_group_rule.port_range_max)), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'port_range_max': self._security_group_rule.port_range_max, 'port_range_min': self._security_group_rule.port_range_min, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_proto_option(self):
        self._setup_security_group_rule({'protocol': 'icmp', 'remote_ip_prefix': '10.0.2.0/24'})
        arglist = ['--proto', self._security_group_rule.protocol, '--remote-ip', self._security_group_rule.remote_ip_prefix, self._security_group.id]
        verifylist = [('proto', self._security_group_rule.protocol), ('protocol', None), ('remote_ip', self._security_group_rule.remote_ip_prefix), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_protocol_any(self):
        self._setup_security_group_rule({'protocol': None, 'remote_ip_prefix': '10.0.2.0/24'})
        arglist = ['--proto', 'any', '--remote-ip', self._security_group_rule.remote_ip_prefix, self._security_group.id]
        verifylist = [('proto', 'any'), ('protocol', None), ('remote_ip', self._security_group_rule.remote_ip_prefix), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_remote_address_group(self):
        self._setup_security_group_rule({'protocol': 'icmp', 'remote_address_group_id': self._address_group.id})
        arglist = ['--protocol', 'icmp', '--remote-address-group', self._address_group.name, self._security_group.id]
        verifylist = [('remote_address_group', self._address_group.name), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'protocol': self._security_group_rule.protocol, 'remote_address_group_id': self._address_group.id, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_remote_group(self):
        self._setup_security_group_rule({'protocol': 'tcp', 'port_range_max': 22, 'port_range_min': 22})
        arglist = ['--protocol', 'tcp', '--dst-port', str(self._security_group_rule.port_range_min), '--ingress', '--remote-group', self._security_group.name, self._security_group.id]
        verifylist = [('dst_port', (self._security_group_rule.port_range_min, self._security_group_rule.port_range_max)), ('ingress', True), ('remote_group', self._security_group.name), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'port_range_max': self._security_group_rule.port_range_max, 'port_range_min': self._security_group_rule.port_range_min, 'protocol': self._security_group_rule.protocol, 'remote_group_id': self._security_group.id, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_source_group(self):
        self._setup_security_group_rule({'remote_group_id': self._security_group.id})
        arglist = ['--ingress', '--remote-group', self._security_group.name, self._security_group.id]
        verifylist = [('ingress', True), ('remote_group', self._security_group.name), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'protocol': self._security_group_rule.protocol, 'remote_group_id': self._security_group_rule.remote_group_id, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_source_ip(self):
        self._setup_security_group_rule({'protocol': 'icmp', 'remote_ip_prefix': '10.0.2.0/24'})
        arglist = ['--protocol', self._security_group_rule.protocol, '--remote-ip', self._security_group_rule.remote_ip_prefix, self._security_group.id]
        verifylist = [('protocol', self._security_group_rule.protocol), ('remote_ip', self._security_group_rule.remote_ip_prefix), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_remote_ip(self):
        self._setup_security_group_rule({'protocol': 'icmp', 'remote_ip_prefix': '10.0.2.0/24'})
        arglist = ['--protocol', self._security_group_rule.protocol, '--remote-ip', self._security_group_rule.remote_ip_prefix, self._security_group.id]
        verifylist = [('protocol', self._security_group_rule.protocol), ('remote_ip', self._security_group_rule.remote_ip_prefix), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_network_options(self):
        self._setup_security_group_rule({'direction': 'egress', 'ether_type': 'IPv6', 'port_range_max': 443, 'port_range_min': 443, 'protocol': '6', 'remote_group_id': None, 'remote_ip_prefix': '::/0'})
        arglist = ['--dst-port', str(self._security_group_rule.port_range_min), '--egress', '--ethertype', self._security_group_rule.ether_type, '--project', self.project.name, '--project-domain', self.domain.name, '--protocol', self._security_group_rule.protocol, self._security_group.id]
        verifylist = [('dst_port', (self._security_group_rule.port_range_min, self._security_group_rule.port_range_max)), ('egress', True), ('ethertype', self._security_group_rule.ether_type), ('project', self.project.name), ('project_domain', self.domain.name), ('protocol', self._security_group_rule.protocol), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'port_range_max': self._security_group_rule.port_range_max, 'port_range_min': self._security_group_rule.port_range_min, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id, 'project_id': self.project.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_tcp_with_icmp_type(self):
        arglist = ['--protocol', 'tcp', '--icmp-type', '15', self._security_group.id]
        verifylist = [('protocol', 'tcp'), ('icmp_type', 15), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_icmp_code(self):
        arglist = ['--protocol', '1', '--icmp-code', '1', self._security_group.id]
        verifylist = [('protocol', '1'), ('icmp_code', 1), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_icmp_code_zero(self):
        self._setup_security_group_rule({'port_range_min': 15, 'port_range_max': 0, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
        arglist = ['--protocol', self._security_group_rule.protocol, '--icmp-type', str(self._security_group_rule.port_range_min), '--icmp-code', str(self._security_group_rule.port_range_max), self._security_group.id]
        verifylist = [('protocol', self._security_group_rule.protocol), ('icmp_code', self._security_group_rule.port_range_max), ('icmp_type', self._security_group_rule.port_range_min), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_icmp_code_greater_than_zero(self):
        self._setup_security_group_rule({'port_range_min': 15, 'port_range_max': 18, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
        arglist = ['--protocol', self._security_group_rule.protocol, '--icmp-type', str(self._security_group_rule.port_range_min), '--icmp-code', str(self._security_group_rule.port_range_max), self._security_group.id]
        verifylist = [('protocol', self._security_group_rule.protocol), ('icmp_type', self._security_group_rule.port_range_min), ('icmp_code', self._security_group_rule.port_range_max), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_icmp_code_negative_value(self):
        self._setup_security_group_rule({'port_range_min': 15, 'port_range_max': None, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
        arglist = ['--protocol', self._security_group_rule.protocol, '--icmp-type', str(self._security_group_rule.port_range_min), '--icmp-code', '-2', self._security_group.id]
        verifylist = [('protocol', self._security_group_rule.protocol), ('icmp_type', self._security_group_rule.port_range_min), ('icmp_code', -2), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_icmp_type(self):
        self._setup_security_group_rule({'port_range_min': 15, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
        arglist = ['--icmp-type', str(self._security_group_rule.port_range_min), '--protocol', self._security_group_rule.protocol, self._security_group.id]
        verifylist = [('dst_port', None), ('icmp_type', self._security_group_rule.port_range_min), ('icmp_code', None), ('protocol', self._security_group_rule.protocol), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'port_range_min': self._security_group_rule.port_range_min, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_icmp_type_zero(self):
        self._setup_security_group_rule({'port_range_min': 0, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
        arglist = ['--icmp-type', str(self._security_group_rule.port_range_min), '--protocol', self._security_group_rule.protocol, self._security_group.id]
        verifylist = [('dst_port', None), ('icmp_type', self._security_group_rule.port_range_min), ('icmp_code', None), ('protocol', self._security_group_rule.protocol), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'port_range_min': self._security_group_rule.port_range_min, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_icmp_type_greater_than_zero(self):
        self._setup_security_group_rule({'port_range_min': 13, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
        arglist = ['--icmp-type', str(self._security_group_rule.port_range_min), '--protocol', self._security_group_rule.protocol, self._security_group.id]
        verifylist = [('dst_port', None), ('icmp_type', self._security_group_rule.port_range_min), ('icmp_code', None), ('protocol', self._security_group_rule.protocol), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'port_range_min': self._security_group_rule.port_range_min, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_icmp_type_negative_value(self):
        self._setup_security_group_rule({'port_range_min': None, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
        arglist = ['--icmp-type', '-13', '--protocol', self._security_group_rule.protocol, self._security_group.id]
        verifylist = [('dst_port', None), ('icmp_type', -13), ('icmp_code', None), ('protocol', self._security_group_rule.protocol), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_ipv6_icmp_type_code(self):
        self._setup_security_group_rule({'ether_type': 'IPv6', 'port_range_min': 139, 'port_range_max': 2, 'protocol': 'ipv6-icmp', 'remote_ip_prefix': '::/0'})
        arglist = ['--icmp-type', str(self._security_group_rule.port_range_min), '--icmp-code', str(self._security_group_rule.port_range_max), '--protocol', self._security_group_rule.protocol, self._security_group.id]
        verifylist = [('dst_port', None), ('icmp_type', self._security_group_rule.port_range_min), ('icmp_code', self._security_group_rule.port_range_max), ('protocol', self._security_group_rule.protocol), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'port_range_min': self._security_group_rule.port_range_min, 'port_range_max': self._security_group_rule.port_range_max, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_icmpv6_type(self):
        self._setup_security_group_rule({'ether_type': 'IPv6', 'port_range_min': 139, 'protocol': 'icmpv6', 'remote_ip_prefix': '::/0'})
        arglist = ['--icmp-type', str(self._security_group_rule.port_range_min), '--protocol', self._security_group_rule.protocol, self._security_group.id]
        verifylist = [('dst_port', None), ('icmp_type', self._security_group_rule.port_range_min), ('icmp_code', None), ('protocol', self._security_group_rule.protocol), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'port_range_min': self._security_group_rule.port_range_min, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_create_with_description(self):
        self._setup_security_group_rule({'description': 'Setting SGR'})
        arglist = ['--description', self._security_group_rule.description, self._security_group.id]
        verifylist = [('description', self._security_group_rule.description), ('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_security_group_rule.assert_called_once_with(**{'description': self._security_group_rule.description, 'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)