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
def test_subport_list(self):
    arglist = ['--trunk', self._trunk['name']]
    verifylist = [('trunk', self._trunk['name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.get_trunk_subports.assert_called_once_with(self._trunk)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))