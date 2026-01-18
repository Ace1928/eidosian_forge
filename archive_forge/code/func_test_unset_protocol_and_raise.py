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
def test_unset_protocol_and_raise(self):
    self.networkclient.update_firewall_rule.side_effect = Exception
    target = self.resource['id']
    arglist = [target, '--protocol']
    verifylist = [(self.res, target), ('protocol', False)]
    self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)