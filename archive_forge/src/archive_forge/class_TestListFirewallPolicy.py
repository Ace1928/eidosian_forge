import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
class TestListFirewallPolicy(TestFirewallPolicy, common.TestListFWaaS):

    def setUp(self):
        super(TestListFirewallPolicy, self).setUp()
        self.networkclient.firewall_policies = mock.Mock(return_value=[_fwp])
        self.mocked = self.networkclient.firewall_policies
        self.cmd = firewallpolicy.ListFirewallPolicy(self.app, self.namespace)