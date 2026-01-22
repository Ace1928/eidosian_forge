from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch('openstackclient.api.compute_v2.APIv2.network_list')
class TestListNetworkCompute(compute_fakes.TestComputev2):
    _networks = compute_fakes.create_networks(count=3)
    columns = ('ID', 'Name', 'Subnet')
    data = []
    for net in _networks:
        data.append((net['id'], net['label'], net['cidr']))

    def setUp(self):
        super(TestListNetworkCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        self.cmd = network.ListNetwork(self.app, None)

    def test_network_list_no_options(self, net_mock):
        net_mock.return_value = self._networks
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        net_mock.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))