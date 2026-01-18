from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_subnet_list_subnetpool_by_id(self):
    subnet_pool = network_fakes.FakeSubnetPool.create_one_subnet_pool()
    subnet = network_fakes.FakeSubnet.create_one_subnet({'subnetpool_id': subnet_pool.id})
    self.network_client.find_network = mock.Mock(return_value=subnet)
    self.network_client.find_subnet_pool = mock.Mock(return_value=subnet_pool)
    arglist = ['--subnet-pool', subnet_pool.id]
    verifylist = [('subnet_pool', subnet_pool.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    filters = {'subnetpool_id': subnet_pool.id}
    self.network_client.subnets.assert_called_once_with(**filters)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))