from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.osc.v2 import share_network_subnets as osc_share_subnets
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_network_subnet_create_metadata_api_version_exception(self):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.77')
    arglist = [self.share_network.id, '--property', 'Manila=zorilla']
    verifylist = [('share_network', self.share_network.id), ('property', {'Manila': 'zorilla'})]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)