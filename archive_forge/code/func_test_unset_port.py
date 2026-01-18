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
def test_unset_port(self):
    target = self.resource['id']
    port = 'old_port'

    def _mock_port_fwg(*args, **kwargs):
        if self.networkclient.find_firewall_group.call_count in [1, 2]:
            self.networkclient.find_firewall_group.assert_called_with(target)
            return {'id': args[0], 'ports': _fwg['ports']}
        if self.networkclient.find_port.call_count == 2:
            self.networkclient.find_port.assert_called_with(target)
            return {'ports': _fwg['ports']}
        if self.networkclient.find_port.call_count == 3:
            self.networkclient.find_port.assert_called_with(port)
            return {'id': args[0]}
    self.networkclient.find_firewall_group.side_effect = _mock_port_fwg
    self.networkclient.find_port.side_effect = _mock_port_fwg
    arglist = [target, '--port', port]
    verifylist = [(self.res, target), ('port', [port])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.mocked.assert_called_once_with(target, **{'ports': []})
    self.assertIsNone(result)