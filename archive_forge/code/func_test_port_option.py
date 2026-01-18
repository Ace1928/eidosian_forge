from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_port_option(self):
    arglist = [self.floating_ip.id, '--port', self.floating_ip.port_id]
    verifylist = [('floating_ip', self.floating_ip.id), ('port', self.floating_ip.port_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    attrs = {'port_id': self.floating_ip.port_id}
    self.network_client.find_ip.assert_called_once_with(self.floating_ip.id, ignore_missing=False)
    self.network_client.update_ip.assert_called_once_with(self.floating_ip, **attrs)