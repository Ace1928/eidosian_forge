from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.osc.v2 import share_network_subnets as osc_share_subnets
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_unset_share_network_subnet_property_exception(self):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.78')
    arglist = [self.share_network.id, self.share_network_subnet.id, '--property', 'Manila', '--property', 'test']
    verifylist = [('share_network', self.share_network.id), ('share_network_subnet', self.share_network_subnet.id), ('property', ['Manila', 'test'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.share_subnets_mock.delete_metadata.assert_has_calls([mock.call(self.share_network.id, ['Manila'], subresource=self.share_network_subnet.id), mock.call(self.share_network.id, ['test'], subresource=self.share_network_subnet.id)])
    self.share_subnets_mock.delete_metadata.side_effect = exceptions.NotFound
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)