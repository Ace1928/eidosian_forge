from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_remove_flavor_from_service_profile(self):
    arglist = [self.network_flavor.id, self.service_profile.id]
    verifylist = [('flavor', self.network_flavor.id), ('service_profile', self.service_profile.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.network_client.disassociate_flavor_from_service_profile.assert_called_once_with(self.network_flavor, self.service_profile)