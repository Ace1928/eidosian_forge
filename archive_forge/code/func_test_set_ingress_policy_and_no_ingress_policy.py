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
def test_set_ingress_policy_and_no_ingress_policy(self):
    target = self.resource['id']
    arglist = [target, '--ingress-firewall-policy', 'my-ingress', '--no-ingress-firewall-policy']
    verifylist = [(self.res, target), ('ingress_firewall_policy', 'my-ingress'), ('no_ingress_firewall_policy', True)]
    self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)