import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
def test_unset_firewall_rule_matched(self):
    _fwp['firewall_rules'] = ['rule1', 'rule2']
    target = self.resource['id']
    rule = 'rule1'
    arglist = [target, '--firewall-rule', rule]
    verifylist = [(self.res, target), ('firewall_rule', [rule])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    body = {'firewall_rules': ['rule2']}
    self.mocked.assert_called_once_with(target, **body)
    self.assertIsNone(result)
    self.assertEqual(2, self.networkclient.find_firewall_policy.call_count)
    self.assertEqual(1, self.networkclient.find_firewall_rule.call_count)