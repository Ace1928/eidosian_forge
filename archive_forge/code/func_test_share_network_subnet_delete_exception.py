from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.osc.v2 import share_network_subnets as osc_share_subnets
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_network_subnet_delete_exception(self):
    arglist = [self.share_network.id, self.share_network_subnets[0].id]
    verifylist = [('share_network', self.share_network.id), ('share_network_subnet', [self.share_network_subnets[0].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.share_subnets_mock.delete.side_effect = exceptions.CommandError()
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)