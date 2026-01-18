from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_host
def test_volume_host_failover(self):
    arglist = ['--volume-backend', 'backend_test', self.service.host]
    verifylist = [('volume_backend', 'backend_test'), ('host', self.service.host)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.host_mock.failover_host.assert_called_with(self.service.host, 'backend_test')
    self.assertIsNone(result)