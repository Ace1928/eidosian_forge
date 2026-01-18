import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
def test_remove_with_no_firewall_rule(self):
    target = self.resource['id']
    arglist = [target]
    verifylist = [(self.res, target)]
    self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)