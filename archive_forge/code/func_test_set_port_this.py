from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_set_port_this(self):
    arglist = ['--disable', '--no-fixed-ip', '--no-binding-profile', self._port.name]
    verifylist = [('disable', True), ('no_binding_profile', True), ('no_fixed_ip', True), ('port', self._port.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'admin_state_up': False, 'binding:profile': {}, 'fixed_ips': []}
    self.network_client.update_port.assert_called_once_with(self._port, **attrs)
    self.assertIsNone(result)