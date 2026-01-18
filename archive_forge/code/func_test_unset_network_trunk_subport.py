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
def test_unset_network_trunk_subport(self):
    subport = self._trunk['sub_ports'][0]
    arglist = ['--subport', subport['port_id'], self._trunk['name']]
    verifylist = [('trunk', self._trunk['name']), ('unset_subports', [subport['port_id']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.delete_trunk_subports.assert_called_once_with(self._trunk, [{'port_id': subport['port_id']}])
    self.assertIsNone(result)