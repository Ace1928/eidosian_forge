from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_subnet_list_network(self):
    network = network_fakes.create_one_network()
    self.network_client.find_network = mock.Mock(return_value=network)
    arglist = ['--network', network.id]
    verifylist = [('network', network.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    filters = {'network_id': network.id}
    self.network_client.subnets.assert_called_once_with(**filters)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))