from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListNetworkSegment(TestNetworkSegment):
    _network = network_fakes.create_one_network()
    _network_segments = network_fakes.create_network_segments(count=3)
    columns = ('ID', 'Name', 'Network', 'Network Type', 'Segment')
    columns_long = columns + ('Physical Network',)
    data = []
    for _network_segment in _network_segments:
        data.append((_network_segment.id, _network_segment.name, _network_segment.network_id, _network_segment.network_type, _network_segment.segmentation_id))
    data_long = []
    for _network_segment in _network_segments:
        data_long.append((_network_segment.id, _network_segment.name, _network_segment.network_id, _network_segment.network_type, _network_segment.segmentation_id, _network_segment.physical_network))

    def setUp(self):
        super(TestListNetworkSegment, self).setUp()
        self.cmd = network_segment.ListNetworkSegment(self.app, self.namespace)
        self.network_client.find_network = mock.Mock(return_value=self._network)
        self.network_client.segments = mock.Mock(return_value=self._network_segments)

    def test_list_no_option(self):
        arglist = []
        verifylist = [('long', False), ('network', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.segments.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True), ('network', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.segments.assert_called_once_with()
        self.assertEqual(self.columns_long, columns)
        self.assertEqual(self.data_long, list(data))

    def test_list_network(self):
        arglist = ['--network', self._network.id]
        verifylist = [('long', False), ('network', self._network.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.segments.assert_called_once_with(**{'network_id': self._network.id})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))