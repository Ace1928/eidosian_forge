from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch('openstackclient.api.compute_v2.APIv2.network_find')
class TestShowNetworkCompute(compute_fakes.TestComputev2):
    _network = compute_fakes.create_one_network()
    columns = ('bridge', 'bridge_interface', 'broadcast', 'cidr', 'cidr_v6', 'created_at', 'deleted', 'deleted_at', 'dhcp_server', 'dhcp_start', 'dns1', 'dns2', 'enable_dhcp', 'gateway', 'gateway_v6', 'host', 'id', 'injected', 'label', 'mtu', 'multi_host', 'netmask', 'netmask_v6', 'priority', 'project_id', 'rxtx_base', 'share_address', 'updated_at', 'vlan', 'vpn_private_address', 'vpn_public_address', 'vpn_public_port')
    data = (_network['bridge'], _network['bridge_interface'], _network['broadcast'], _network['cidr'], _network['cidr_v6'], _network['created_at'], _network['deleted'], _network['deleted_at'], _network['dhcp_server'], _network['dhcp_start'], _network['dns1'], _network['dns2'], _network['enable_dhcp'], _network['gateway'], _network['gateway_v6'], _network['host'], _network['id'], _network['injected'], _network['label'], _network['mtu'], _network['multi_host'], _network['netmask'], _network['netmask_v6'], _network['priority'], _network['project_id'], _network['rxtx_base'], _network['share_address'], _network['updated_at'], _network['vlan'], _network['vpn_private_address'], _network['vpn_public_address'], _network['vpn_public_port'])

    def setUp(self):
        super(TestShowNetworkCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        self.cmd = network.ShowNetwork(self.app, None)

    def test_show_no_options(self, net_mock):
        net_mock.return_value = self._network
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self, net_mock):
        net_mock.return_value = self._network
        arglist = [self._network['label']]
        verifylist = [('network', self._network['label'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        net_mock.assert_called_once_with(self._network['label'])
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)