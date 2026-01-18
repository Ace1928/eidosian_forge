from unittest import mock
from unittest.mock import call
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import service
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
@mock.patch.object(sdk_utils, 'supports_microversion')
def test_service_set_2_53_disable_down(self, sm_mock):
    sm_mock.return_value = True
    arglist = ['--disable', '--down', self.service.host, self.service.binary]
    verifylist = [('disable', True), ('down', True), ('host', self.service.host), ('service', self.service.binary)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    service_id = '339478d0-0b95-4a94-be63-d5be05dfeb1c'
    self.compute_sdk_client.services.return_value = [mock.Mock(id=service_id)]
    result = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.disable_service.assert_called_once_with(service_id, self.service.host, self.service.binary, None)
    self.compute_sdk_client.update_service_forced_down.assert_called_once_with(service_id, self.service.host, self.service.binary, True)
    self.assertIsNone(result)