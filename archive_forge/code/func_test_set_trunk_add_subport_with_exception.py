import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
import testtools
from openstackclient.network.v2 import network_trunk
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_set_trunk_add_subport_with_exception(self):
    arglist = ['--subport', 'port=invalid_subport', self._trunk['name']]
    verifylist = [('trunk', self._trunk['name']), ('set_subports', [{'port': 'invalid_subport'}])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.network_client.add_trunk_subports = mock.Mock(side_effect=exceptions.CommandError)
    self.network_client.find_port = mock.Mock(return_value={'id': 'invalid_subport'})
    with testtools.ExpectedException(exceptions.CommandError) as e:
        self.cmd.take_action(parsed_args)
        self.assertEqual("Failed to add subports to trunk '%s': " % self._trunk['name'], str(e))
    self.network_client.update_trunk.assert_called_once_with(self._trunk)
    self.network_client.add_trunk_subports.assert_called_once_with(self._trunk, [{'port_id': 'invalid_subport'}])