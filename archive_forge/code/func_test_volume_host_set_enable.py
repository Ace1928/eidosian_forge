from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_host
def test_volume_host_set_enable(self):
    arglist = ['--enable', self.service.host]
    verifylist = [('enable', True), ('host', self.service.host)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.host_mock.thaw_host.assert_called_with(self.service.host)
    self.host_mock.freeze_host.assert_not_called()
    self.assertIsNone(result)