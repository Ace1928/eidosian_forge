from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_overwrite_options(self):
    _testsubnet = network_fakes.FakeSubnet.create_one_subnet({'host_routes': [{'destination': '10.20.20.0/24', 'nexthop': '10.20.20.1'}], 'allocation_pools': [{'start': '8.8.8.200', 'end': '8.8.8.250'}], 'dns_nameservers': ['10.0.0.1']})
    self.network_client.find_subnet = mock.Mock(return_value=_testsubnet)
    arglist = ['--host-route', 'destination=10.30.30.30/24,gateway=10.30.30.1', '--no-host-route', '--allocation-pool', 'start=8.8.8.100,end=8.8.8.150', '--no-allocation-pool', '--dns-nameserver', '10.1.10.1', '--no-dns-nameservers', _testsubnet.name]
    verifylist = [('host_routes', [{'destination': '10.30.30.30/24', 'gateway': '10.30.30.1'}]), ('allocation_pools', [{'start': '8.8.8.100', 'end': '8.8.8.150'}]), ('dns_nameservers', ['10.1.10.1']), ('no_dns_nameservers', True), ('no_host_route', True), ('no_allocation_pool', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'host_routes': [{'destination': '10.30.30.30/24', 'nexthop': '10.30.30.1'}], 'allocation_pools': [{'start': '8.8.8.100', 'end': '8.8.8.150'}], 'dns_nameservers': ['10.1.10.1']}
    self.network_client.update_subnet.assert_called_once_with(_testsubnet, **attrs)
    self.assertIsNone(result)