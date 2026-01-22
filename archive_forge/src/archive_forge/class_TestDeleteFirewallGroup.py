import copy
import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallgroup
from neutronclient.osc.v2 import utils as v2_utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
class TestDeleteFirewallGroup(TestFirewallGroup, common.TestDeleteFWaaS):

    def setUp(self):
        super(TestDeleteFirewallGroup, self).setUp()
        self.networkclient.delete_firewall_group = mock.Mock()
        self.mocked = self.networkclient.delete_firewall_group
        self.cmd = firewallgroup.DeleteFirewallGroup(self.app, self.namespace)