from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip_port_forwarding
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateFloatingIPPortForwarding(TestFloatingIPPortForwarding):

    def setUp(self):
        super(TestCreateFloatingIPPortForwarding, self).setUp()
        self.new_port_forwarding = network_fakes.FakeFloatingIPPortForwarding.create_one_port_forwarding(attrs={'internal_port_id': self.port.id, 'floatingip_id': self.floating_ip.id})
        self.new_port_forwarding_with_ranges = network_fakes.FakeFloatingIPPortForwarding.create_one_port_forwarding(use_range=True, attrs={'internal_port_id': self.port.id, 'floatingip_id': self.floating_ip.id})
        self.network_client.create_floating_ip_port_forwarding = mock.Mock(return_value=self.new_port_forwarding)
        self.network_client.find_ip = mock.Mock(return_value=self.floating_ip)
        self.cmd = floating_ip_port_forwarding.CreateFloatingIPPortForwarding(self.app, self.namespace)
        self.columns = ('description', 'external_port', 'external_port_range', 'floatingip_id', 'id', 'internal_ip_address', 'internal_port', 'internal_port_id', 'internal_port_range', 'protocol')
        self.data = (self.new_port_forwarding.description, self.new_port_forwarding.external_port, self.new_port_forwarding.external_port_range, self.new_port_forwarding.floatingip_id, self.new_port_forwarding.id, self.new_port_forwarding.internal_ip_address, self.new_port_forwarding.internal_port, self.new_port_forwarding.internal_port_id, self.new_port_forwarding.internal_port_range, self.new_port_forwarding.protocol)

    def test_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_all_options_with_range(self):
        arglist = ['--port', self.new_port_forwarding_with_ranges.internal_port_id, '--internal-protocol-port', self.new_port_forwarding_with_ranges.internal_port_range, '--external-protocol-port', self.new_port_forwarding_with_ranges.external_port_range, '--protocol', self.new_port_forwarding_with_ranges.protocol, self.new_port_forwarding_with_ranges.floatingip_id, '--internal-ip-address', self.new_port_forwarding_with_ranges.internal_ip_address, '--description', self.new_port_forwarding_with_ranges.description]
        verifylist = [('port', self.new_port_forwarding_with_ranges.internal_port_id), ('internal_protocol_port', self.new_port_forwarding_with_ranges.internal_port_range), ('external_protocol_port', self.new_port_forwarding_with_ranges.external_port_range), ('protocol', self.new_port_forwarding_with_ranges.protocol), ('floating_ip', self.new_port_forwarding_with_ranges.floatingip_id), ('internal_ip_address', self.new_port_forwarding_with_ranges.internal_ip_address), ('description', self.new_port_forwarding_with_ranges.description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_floating_ip_port_forwarding.assert_called_once_with(self.new_port_forwarding.floatingip_id, **{'external_port_range': self.new_port_forwarding_with_ranges.external_port_range, 'internal_ip_address': self.new_port_forwarding_with_ranges.internal_ip_address, 'internal_port_range': self.new_port_forwarding_with_ranges.internal_port_range, 'internal_port_id': self.new_port_forwarding_with_ranges.internal_port_id, 'protocol': self.new_port_forwarding_with_ranges.protocol, 'description': self.new_port_forwarding_with_ranges.description})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_all_options_with_range_invalid_port_exception(self):
        invalid_port_range = '999999:999999'
        arglist = ['--port', self.new_port_forwarding_with_ranges.internal_port_id, '--internal-protocol-port', invalid_port_range, '--external-protocol-port', invalid_port_range, '--protocol', self.new_port_forwarding_with_ranges.protocol, self.new_port_forwarding_with_ranges.floatingip_id, '--internal-ip-address', self.new_port_forwarding_with_ranges.internal_ip_address, '--description', self.new_port_forwarding_with_ranges.description]
        verifylist = [('port', self.new_port_forwarding_with_ranges.internal_port_id), ('internal_protocol_port', invalid_port_range), ('external_protocol_port', invalid_port_range), ('protocol', self.new_port_forwarding_with_ranges.protocol), ('floating_ip', self.new_port_forwarding_with_ranges.floatingip_id), ('internal_ip_address', self.new_port_forwarding_with_ranges.internal_ip_address), ('description', self.new_port_forwarding_with_ranges.description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        msg = 'The port number range is <1-65535>'
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual(msg, str(e))
            self.network_client.create_floating_ip_port_forwarding.assert_not_called()

    def test_create_all_options_with_invalid_range_exception(self):
        invalid_port_range = '80:70'
        arglist = ['--port', self.new_port_forwarding_with_ranges.internal_port_id, '--internal-protocol-port', invalid_port_range, '--external-protocol-port', invalid_port_range, '--protocol', self.new_port_forwarding_with_ranges.protocol, self.new_port_forwarding_with_ranges.floatingip_id, '--internal-ip-address', self.new_port_forwarding_with_ranges.internal_ip_address, '--description', self.new_port_forwarding_with_ranges.description]
        verifylist = [('port', self.new_port_forwarding_with_ranges.internal_port_id), ('internal_protocol_port', invalid_port_range), ('external_protocol_port', invalid_port_range), ('protocol', self.new_port_forwarding_with_ranges.protocol), ('floating_ip', self.new_port_forwarding_with_ranges.floatingip_id), ('internal_ip_address', self.new_port_forwarding_with_ranges.internal_ip_address), ('description', self.new_port_forwarding_with_ranges.description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        msg = 'The last number in port range must be greater or equal to the first'
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual(msg, str(e))
            self.network_client.create_floating_ip_port_forwarding.assert_not_called()

    def test_create_all_options_with_unmatch_ranges_exception(self):
        internal_range = '80:90'
        external_range = '8080:8100'
        arglist = ['--port', self.new_port_forwarding_with_ranges.internal_port_id, '--internal-protocol-port', internal_range, '--external-protocol-port', external_range, '--protocol', self.new_port_forwarding_with_ranges.protocol, self.new_port_forwarding_with_ranges.floatingip_id, '--internal-ip-address', self.new_port_forwarding_with_ranges.internal_ip_address, '--description', self.new_port_forwarding_with_ranges.description]
        verifylist = [('port', self.new_port_forwarding_with_ranges.internal_port_id), ('internal_protocol_port', internal_range), ('external_protocol_port', external_range), ('protocol', self.new_port_forwarding_with_ranges.protocol), ('floating_ip', self.new_port_forwarding_with_ranges.floatingip_id), ('internal_ip_address', self.new_port_forwarding_with_ranges.internal_ip_address), ('description', self.new_port_forwarding_with_ranges.description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        msg = 'The relation between internal and external ports does not match the pattern 1:N and N:N'
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual(msg, str(e))
            self.network_client.create_floating_ip_port_forwarding.assert_not_called()

    def test_create_all_options(self):
        arglist = ['--port', self.new_port_forwarding.internal_port_id, '--internal-protocol-port', str(self.new_port_forwarding.internal_port), '--external-protocol-port', str(self.new_port_forwarding.external_port), '--protocol', self.new_port_forwarding.protocol, self.new_port_forwarding.floatingip_id, '--internal-ip-address', self.new_port_forwarding.internal_ip_address, '--description', self.new_port_forwarding.description]
        verifylist = [('port', self.new_port_forwarding.internal_port_id), ('internal_protocol_port', str(self.new_port_forwarding.internal_port)), ('external_protocol_port', str(self.new_port_forwarding.external_port)), ('protocol', self.new_port_forwarding.protocol), ('floating_ip', self.new_port_forwarding.floatingip_id), ('internal_ip_address', self.new_port_forwarding.internal_ip_address), ('description', self.new_port_forwarding.description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_floating_ip_port_forwarding.assert_called_once_with(self.new_port_forwarding.floatingip_id, **{'external_port': self.new_port_forwarding.external_port, 'internal_ip_address': self.new_port_forwarding.internal_ip_address, 'internal_port': self.new_port_forwarding.internal_port, 'internal_port_id': self.new_port_forwarding.internal_port_id, 'protocol': self.new_port_forwarding.protocol, 'description': self.new_port_forwarding.description})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)