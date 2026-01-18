from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_segment(self):
    _net = network_fakes.create_one_network()
    _segment = network_fakes.create_one_network_segment(attrs={'network_id': _net.id})
    _subnet = network_fakes.FakeSubnet.create_one_subnet({'host_routes': [{'destination': '10.20.20.0/24', 'nexthop': '10.20.20.1'}], 'allocation_pools': [{'start': '8.8.8.200', 'end': '8.8.8.250'}], 'dns_nameservers': ['10.0.0.1'], 'network_id': _net.id, 'segment_id': None})
    self.network_client.find_subnet = mock.Mock(return_value=_subnet)
    self.network_client.find_segment = mock.Mock(return_value=_segment)
    arglist = ['--network-segment', _segment.id, _subnet.name]
    verifylist = [('network_segment', _segment.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'segment_id': _segment.id}
    self.network_client.update_subnet.assert_called_once_with(_subnet, **attrs)
    self.network_client.update_subnet.assert_called_with(_subnet, **attrs)
    self.assertIsNone(result)