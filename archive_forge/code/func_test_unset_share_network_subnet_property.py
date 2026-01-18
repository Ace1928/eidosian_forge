from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.osc.v2 import share_network_subnets as osc_share_subnets
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_unset_share_network_subnet_property(self):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.78')
    arglist = [self.share_network.id, self.share_network_subnet.id, '--property', 'Manila']
    verifylist = [('share_network', self.share_network.id), ('share_network_subnet', self.share_network_subnet.id), ('property', ['Manila'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.share_subnets_mock.delete_metadata.assert_called_once_with(self.share_network.id, ['Manila'], subresource=self.share_network_subnet.id)