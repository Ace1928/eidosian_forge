from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_non_append_options(self):
    arglist = ['--description', 'new_description', '--dhcp', '--gateway', self._subnet.gateway_ip, self._subnet.name]
    verifylist = [('description', 'new_description'), ('dhcp', True), ('gateway', self._subnet.gateway_ip), ('subnet', self._subnet.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'enable_dhcp': True, 'gateway_ip': self._subnet.gateway_ip, 'description': 'new_description'}
    self.network_client.update_subnet.assert_called_with(self._subnet, **attrs)
    self.assertIsNone(result)