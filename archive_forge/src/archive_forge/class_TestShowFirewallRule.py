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
class TestShowFirewallRule(TestFirewallRule, common.TestShowFWaaS):

    def setUp(self):
        super(TestShowFirewallRule, self).setUp()
        self.networkclient.get_firewall_rule = mock.Mock(return_value=_fwr)
        self.mocked = self.networkclient.get_firewall_rule
        self.cmd = firewallrule.ShowFirewallRule(self.app, self.namespace)