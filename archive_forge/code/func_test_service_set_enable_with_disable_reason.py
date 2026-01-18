from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import service
def test_service_set_enable_with_disable_reason(self):
    reason = 'earthquake'
    arglist = ['--enable', '--disable-reason', reason, self.service.host, self.service.binary]
    verifylist = [('enable', True), ('disable_reason', reason), ('host', self.service.host), ('service', self.service.binary)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('Cannot specify option --disable-reason without --disable specified.', str(e))