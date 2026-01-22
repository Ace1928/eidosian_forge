import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
class TestSetFirewallPolicy(TestFirewallPolicy, common.TestSetFWaaS):

    def setUp(self):
        super(TestSetFirewallPolicy, self).setUp()
        self.networkclient.update_firewall_policy = mock.Mock(return_value=_fwp)
        self.mocked = self.networkclient.update_firewall_policy

        def _mock_find_rule(*args, **kwargs):
            return {'id': args[0]}

        def _mock_find_policy(*args, **kwargs):
            return {'id': args[0], 'firewall_rules': _fwp['firewall_rules']}
        self.networkclient.find_firewall_policy.side_effect = _mock_find_policy
        self.networkclient.find_firewall_rule.side_effect = _mock_find_rule
        self.cmd = firewallpolicy.SetFirewallPolicy(self.app, self.namespace)

    def test_set_rules(self):
        target = self.resource['id']
        rule1 = 'new_rule1'
        rule2 = 'new_rule2'
        arglist = [target, '--firewall-rule', rule1, '--firewall-rule', rule2]
        verifylist = [(self.res, target), ('firewall_rule', [rule1, rule2])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        expect = _fwp['firewall_rules'] + [rule1, rule2]
        body = {'firewall_rules': expect}
        self.mocked.assert_called_once_with(target, **body)
        self.assertEqual(2, self.networkclient.find_firewall_rule.call_count)
        self.assertEqual(2, self.networkclient.find_firewall_policy.call_count)
        self.assertIsNone(result)

    def test_set_no_rules(self):
        target = self.resource['id']
        arglist = [target, '--no-firewall-rule']
        verifylist = [(self.res, target), ('no_firewall_rule', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        body = {'firewall_rules': []}
        self.mocked.assert_called_once_with(target, **body)
        self.assertIsNone(result)

    def test_set_rules_and_no_rules(self):
        target = self.resource['id']
        rule1 = 'rule1'
        arglist = [target, '--firewall-rule', rule1, '--no-firewall-rule']
        verifylist = [(self.res, target), ('firewall_rule', [rule1]), ('no_firewall_rule', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        body = {'firewall_rules': [rule1]}
        self.mocked.assert_called_once_with(target, **body)
        self.assertEqual(1, self.networkclient.find_firewall_rule.call_count)
        self.assertEqual(1, self.networkclient.find_firewall_policy.call_count)
        self.assertIsNone(result)

    def test_set_audited(self):
        target = self.resource['id']
        arglist = [target, '--audited']
        verifylist = [(self.res, target), ('audited', True)]
        body = {'audited': True}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **body)
        self.assertIsNone(result)

    def test_set_no_audited(self):
        target = self.resource['id']
        arglist = [target, '--no-audited']
        verifylist = [(self.res, target), ('no_audited', True)]
        body = {'audited': False}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, **body)
        self.assertIsNone(result)

    def test_set_audited_and_no_audited(self):
        target = self.resource['id']
        arglist = [target, '--audited', '--no-audited']
        verifylist = [(self.res, target), ('audited', True), ('no_audited', True)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_and_raises(self):
        self.networkclient.update_firewall_policy = mock.Mock(side_effect=Exception)
        target = self.resource['id']
        arglist = [target, '--name', 'my-name']
        verifylist = [(self.res, target), ('name', 'my-name')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)