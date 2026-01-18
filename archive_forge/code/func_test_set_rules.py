import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
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