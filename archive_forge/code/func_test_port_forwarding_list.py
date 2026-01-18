from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip_port_forwarding
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_port_forwarding_list(self):
    arglist = [self.floating_ip.id]
    verifylist = [('floating_ip', self.floating_ip.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.floating_ip_port_forwardings.assert_called_once_with(self.floating_ip, **{})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))