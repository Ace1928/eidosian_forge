from unittest import mock
from unittest.mock import call
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import service
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
@mock.patch.object(sdk_utils, 'supports_microversion')
def test_service_set_state_up(self, sm_mock):
    sm_mock.side_effect = [False, True]
    arglist = ['--up', self.service.host, self.service.binary]
    verifylist = [('up', True), ('host', self.service.host), ('service', self.service.binary)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.update_service_forced_down.assert_called_once_with(None, self.service.host, self.service.binary, False)
    self.assertNotCalled(self.compute_sdk_client.enable_service)
    self.assertNotCalled(self.compute_sdk_client.disable_service)
    self.assertIsNone(result)