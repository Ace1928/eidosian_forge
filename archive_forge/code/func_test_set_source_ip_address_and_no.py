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
def test_set_source_ip_address_and_no(self):
    target = self.resource['id']
    arglist = [target, '--source-ip-address', '192.168.1.0/24', '--no-source-ip-address']
    verifylist = [(self.res, target), ('source_ip_address', '192.168.1.0/24'), ('no_source_ip_address', True)]
    self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)