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
def test_set_source_ip_address(self):
    target = self.resource['id']
    src_ip = '192.192.192.192'
    arglist = [target, '--source-ip-address', src_ip]
    verifylist = [(self.res, target), ('source_ip_address', src_ip)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.mocked.assert_called_once_with(target, **{'source_ip_address': src_ip})
    self.assertIsNone(result)