from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_unset_subnet_params(self):
    arglist = ['--dns-nameserver', '8.8.8.8', '--host-route', 'destination=10.30.30.30/24,gateway=10.30.30.1', '--allocation-pool', 'start=8.8.8.100,end=8.8.8.150', '--service-type', 'network:router_gateway', '--gateway', self._testsubnet.name]
    verifylist = [('dns_nameservers', ['8.8.8.8']), ('host_routes', [{'destination': '10.30.30.30/24', 'gateway': '10.30.30.1'}]), ('allocation_pools', [{'start': '8.8.8.100', 'end': '8.8.8.150'}]), ('service_types', ['network:router_gateway']), ('gateway', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'dns_nameservers': ['8.8.8.4'], 'host_routes': [{'destination': '10.20.20.0/24', 'nexthop': '10.20.20.1'}], 'allocation_pools': [{'start': '8.8.8.160', 'end': '8.8.8.170'}], 'service_types': ['network:floatingip_agent_gateway'], 'gateway_ip': None}
    self.network_client.update_subnet.assert_called_once_with(self._testsubnet, **attrs)
    self.assertIsNone(result)