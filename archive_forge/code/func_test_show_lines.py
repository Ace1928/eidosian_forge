from unittest import mock
from openstackclient.compute.v2 import console
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils
def test_show_lines(self):
    arglist = ['fake_server', '--lines', '15']
    verifylist = [('server', 'fake_server'), ('lines', 15)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    output = {'output': '1st line\n2nd line'}
    self.compute_sdk_client.get_server_console_output.return_value = output
    self.cmd.take_action(parsed_args)
    self.compute_sdk_client.find_server.assert_called_with(name_or_id='fake_server', ignore_missing=False)
    self.compute_sdk_client.get_server_console_output.assert_called_with(self._server.id, length=15)