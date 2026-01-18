from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import service
def test_service_set_nothing(self):
    arglist = [self.service.host, self.service.binary]
    verifylist = [('host', self.service.host), ('service', self.service.binary)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.service_mock.enable.assert_not_called()
    self.service_mock.disable.assert_not_called()
    self.service_mock.disable_log_reason.assert_not_called()
    self.assertIsNone(result)