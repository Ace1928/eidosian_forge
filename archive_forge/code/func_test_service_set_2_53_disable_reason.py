from unittest import mock
from unittest.mock import call
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import service
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
@mock.patch.object(sdk_utils, 'supports_microversion')
def test_service_set_2_53_disable_reason(self, sm_mock):
    sm_mock.return_value = True
    reason = 'earthquake'
    arglist = ['--disable', '--disable-reason', reason, self.service.host, self.service.binary]
    verifylist = [('disable', True), ('disable_reason', reason), ('host', self.service.host), ('service', self.service.binary)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    service_id = '339478d0-0b95-4a94-be63-d5be05dfeb1c'
    self.compute_sdk_client.services.return_value = [mock.Mock(id=service_id)]
    result = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.disable_service.assert_called_once_with(service_id, self.service.host, self.service.binary, reason)
    self.assertIsNone(result)