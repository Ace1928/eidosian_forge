from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_replicas as osc_share_replicas
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_replica_create_share_network(self):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.72')
    arglist = [self.share.id, '--availability-zone', self.share.availability_zone, '--share-network', self.share.share_network_id]
    verifylist = [('share', self.share.id), ('availability_zone', self.share.availability_zone), ('share_network', self.share.share_network_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    if self.share.share_network_id:
        self.replicas_mock.create.assert_called_with(share=self.share, availability_zone=self.share.availability_zone, share_network=self.share.share_network_id)
    else:
        self.replicas_mock.create.assert_called_with(share=self.share, availability_zone=self.share.availability_zone)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)