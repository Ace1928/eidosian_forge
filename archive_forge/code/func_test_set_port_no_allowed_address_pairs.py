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
def test_set_port_no_allowed_address_pairs(self):
    arglist = ['--no-allowed-address', self._port.name]
    verifylist = [('no_allowed_address_pair', True), ('port', self._port.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'allowed_address_pairs': []}
    self.network_client.update_port.assert_called_once_with(self._port, **attrs)
    self.assertIsNone(result)