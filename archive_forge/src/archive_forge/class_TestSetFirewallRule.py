import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallrule
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
class TestSetFirewallRule(TestFirewallRule, common.TestSetFWaaS):

    def setUp(self):
        super(TestSetFirewallRule, self).setUp()
        self.networkclient.update_firewall_rule = mock.Mock(return_value=_fwr)
        self.mocked = self.networkclient.update_firewall_rule

        def _mock_find_rule(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_firewall_rule.side_effect = _mock_find_rule
        self.cmd = firewallrule.SetFirewallRule(self.app, self.namespace)

    def test_set_protocol_with_any(self):
        target = self.resource['id']
        protocol = 'any'
        arglist = [target, '--protocol', protocol]
        verifylist = [(self.res, target), ('protocol', protocol)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'protocol': None})
        self.assertIsNone(result)

    def test_set_protocol_with_udp(self):
        target = self.resource['id']
        protocol = 'udp'
        arglist = [target, '--protocol', protocol]
        verifylist = [(self.res, target), ('protocol', protocol)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'protocol': protocol})
        self.assertIsNone(result)

    def test_set_source_ip_address(self):
        target = self.resource['id']
        src_ip = '192.192.192.192'
        arglist = [target, '--source-ip-address', src_ip]
        verifylist = [(self.res, target), ('source_ip_address', src_ip)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'source_ip_address': src_ip})
        self.assertIsNone(result)

    def test_set_source_port(self):
        target = self.resource['id']
        src_port = '32678'
        arglist = [target, '--source-port', src_port]
        verifylist = [(self.res, target), ('source_port', src_port)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'source_port': src_port})
        self.assertIsNone(result)

    def test_set_destination_ip_address(self):
        target = self.resource['id']
        dst_ip = '0.1.0.1'
        arglist = [target, '--destination-ip-address', dst_ip]
        verifylist = [(self.res, target), ('destination_ip_address', dst_ip)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'destination_ip_address': dst_ip})
        self.assertIsNone(result)

    def test_set_destination_port(self):
        target = self.resource['id']
        dst_port = '65432'
        arglist = [target, '--destination-port', dst_port]
        verifylist = [(self.res, target), ('destination_port', dst_port)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'destination_port': dst_port})
        self.assertIsNone(result)

    def test_set_enable_rule(self):
        target = self.resource['id']
        arglist = [target, '--enable-rule']
        verifylist = [(self.res, target), ('enable_rule', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'enabled': True})
        self.assertIsNone(result)

    def test_set_disable_rule(self):
        target = self.resource['id']
        arglist = [target, '--disable-rule']
        verifylist = [(self.res, target), ('disable_rule', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'enabled': False})
        self.assertIsNone(result)

    def test_set_action(self):
        target = self.resource['id']
        action = 'reject'
        arglist = [target, '--action', action]
        verifylist = [(self.res, target), ('action', action)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'action': action})
        self.assertIsNone(result)

    def test_set_enable_rule_and_disable_rule(self):
        target = self.resource['id']
        arglist = [target, '--enable-rule', '--disable-rule']
        verifylist = [(self.res, target), ('enable_rule', True), ('disable_rule', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_no_source_ip_address(self):
        target = self.resource['id']
        arglist = [target, '--no-source-ip-address']
        verifylist = [(self.res, target), ('no_source_ip_address', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'source_ip_address': None})
        self.assertIsNone(result)

    def test_set_no_source_port(self):
        target = self.resource['id']
        arglist = [target, '--no-source-port']
        verifylist = [(self.res, target), ('no_source_port', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'source_port': None})
        self.assertIsNone(result)

    def test_set_no_destination_ip_address(self):
        target = self.resource['id']
        arglist = [target, '--no-destination-ip-address']
        verifylist = [(self.res, target), ('no_destination_ip_address', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'destination_ip_address': None})
        self.assertIsNone(result)

    def test_set_no_destination_port(self):
        target = self.resource['id']
        arglist = [target, '--no-destination-port']
        verifylist = [(self.res, target), ('no_destination_port', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'destination_port': None})
        self.assertIsNone(result)

    def test_set_source_ip_address_and_no(self):
        target = self.resource['id']
        arglist = [target, '--source-ip-address', '192.168.1.0/24', '--no-source-ip-address']
        verifylist = [(self.res, target), ('source_ip_address', '192.168.1.0/24'), ('no_source_ip_address', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_destination_ip_address_and_no(self):
        target = self.resource['id']
        arglist = [target, '--destination-ip-address', '192.168.2.0/24', '--no-destination-ip-address']
        verifylist = [(self.res, target), ('destination_ip_address', '192.168.2.0/24'), ('no_destination_ip_address', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_source_port_and_no(self):
        target = self.resource['id']
        arglist = [target, '--source-port', '1:12345', '--no-source-port']
        verifylist = [(self.res, target), ('source_port', '1:12345'), ('no_source_port', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_destination_port_and_no(self):
        target = self.resource['id']
        arglist = [target, '--destination-port', '1:54321', '--no-destination-port']
        verifylist = [(self.res, target), ('destination_port', '1:54321'), ('no_destination_port', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_and_raises(self):
        self.networkclient.update_firewall_rule = mock.Mock(side_effect=Exception)
        target = self.resource['id']
        arglist = [target, '--name', 'my-name']
        verifylist = [(self.res, target), ('name', 'my-name')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_set_no_destination_fwg(self):
        target = self.resource['id']
        arglist = [target, '--no-destination-firewall-group']
        verifylist = [(self.res, target), ('no_destination_firewall_group', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'destination_firewall_group_id': None})
        self.assertIsNone(result)

    def test_set_no_source_fwg(self):
        target = self.resource['id']
        arglist = [target, '--no-source-firewall-group']
        verifylist = [(self.res, target), ('no_source_firewall_group', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **{'source_firewall_group_id': None})
        self.assertIsNone(result)

    def test_create_with_src_fwg_and_no(self):
        target = self.resource['id']
        fwg = 'my-fwg'
        arglist = [target, '--source-firewall-group', fwg, '--no-source-firewall-group']
        verifylist = [(self.res, target), ('source_firewall_group', fwg), ('no_source_firewall_group', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_with_dst_fwg_and_no(self):
        target = self.resource['id']
        fwg = 'my-fwg'
        arglist = [target, '--destination-firewall-group', fwg, '--no-destination-firewall-group']
        verifylist = [(self.res, target), ('destination_firewall_group', fwg), ('no_destination_firewall_group', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)