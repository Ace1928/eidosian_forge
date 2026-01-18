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
def test_set_and_raises(self):
    self.networkclient.update_firewall_group = mock.Mock(side_effect=Exception)
    target = self.resource['id']
    arglist = [target, '--name', 'my-name']
    verifylist = [(self.res, target), ('name', 'my-name')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)