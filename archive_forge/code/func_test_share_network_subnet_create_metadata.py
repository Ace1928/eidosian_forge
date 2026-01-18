from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.osc.v2 import share_network_subnets as osc_share_subnets
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_network_subnet_create_metadata(self):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.78')
    arglist = [self.share_network.id, '--property', 'Manila=zorilla', '--property', 'Zorilla=manila']
    verifylist = [('share_network', self.share_network.id), ('property', {'Manila': 'zorilla', 'Zorilla': 'manila'})]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.share_subnets_mock.create.assert_called_once_with(neutron_net_id=None, neutron_subnet_id=None, availability_zone=None, share_network_id=self.share_network.id, metadata={'Manila': 'zorilla', 'Zorilla': 'manila'})
    self.assertEqual(set(self.columns), set(columns))
    self.assertCountEqual(self.data, data)