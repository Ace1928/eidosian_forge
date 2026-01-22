from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowSubnet(TestSubnet):
    _subnet = network_fakes.FakeSubnet.create_one_subnet()
    columns = ('allocation_pools', 'cidr', 'description', 'dns_nameservers', 'enable_dhcp', 'gateway_ip', 'host_routes', 'id', 'ip_version', 'ipv6_address_mode', 'ipv6_ra_mode', 'name', 'network_id', 'project_id', 'segment_id', 'service_types', 'subnetpool_id', 'tags')
    data = (subnet_v2.AllocationPoolsColumn(_subnet.allocation_pools), _subnet.cidr, _subnet.description, format_columns.ListColumn(_subnet.dns_nameservers), _subnet.enable_dhcp, _subnet.gateway_ip, subnet_v2.HostRoutesColumn(_subnet.host_routes), _subnet.id, _subnet.ip_version, _subnet.ipv6_address_mode, _subnet.ipv6_ra_mode, _subnet.name, _subnet.network_id, _subnet.project_id, _subnet.segment_id, format_columns.ListColumn(_subnet.service_types), _subnet.subnetpool_id, format_columns.ListColumn(_subnet.tags))

    def setUp(self):
        super(TestShowSubnet, self).setUp()
        self.cmd = subnet_v2.ShowSubnet(self.app, self.namespace)
        self.network_client.find_subnet = mock.Mock(return_value=self._subnet)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self):
        arglist = [self._subnet.name]
        verifylist = [('subnet', self._subnet.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_subnet.assert_called_once_with(self._subnet.name, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)