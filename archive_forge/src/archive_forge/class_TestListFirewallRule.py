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
class TestListFirewallRule(TestFirewallRule):

    def _setup_summary(self, expect=None):
        protocol = (_fwr['protocol'] or 'any').upper()
        src = 'source(port): 192.168.1.0/24(1:11111)'
        dst = 'dest(port): 192.168.2.2(2:22222)'
        action = 'deny'
        if expect:
            if expect.get('protocol'):
                protocol = expect['protocol']
            if expect.get('source_ip_address'):
                src_ip = expect['source_ip_address']
            if expect.get('source_port'):
                src_port = expect['source_port']
            if expect.get('destination_ip_address'):
                dst_ip = expect['destination_ip_address']
            if expect.get('destination_port'):
                dst_port = expect['destination_port']
            if expect.get('action'):
                action = expect['action']
            src = 'source(port): ' + src_ip + '(' + src_port + ')'
            dst = 'dest(port): ' + dst_ip + '(' + dst_port + ')'
        return ',\n '.join([protocol, src, dst, action])

    def setUp(self):
        super(TestListFirewallRule, self).setUp()
        self.cmd = firewallrule.ListFirewallRule(self.app, self.namespace)
        self.short_header = ('ID', 'Name', 'Enabled', 'Summary', 'Firewall Policy')
        summary = self._setup_summary(_fwr)
        self.short_data = (_fwr['id'], _fwr['name'], _fwr['enabled'], summary, _fwr['firewall_policy_id'])
        self.networkclient.firewall_rules = mock.Mock(return_value=[_fwr])
        self.mocked = self.networkclient.firewall_rules

    def test_list_with_long_option(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with()
        self.assertEqual(list(self.headers), headers)

    def test_list_with_no_option(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with()
        self.assertEqual(list(self.short_header), headers)
        self.assertListItemEqual([self.short_data], list(data))