from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment_range
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowNetworkSegmentRange(TestNetworkSegmentRange):
    _network_segment_range = network_fakes.create_one_network_segment_range()
    columns = ('available', 'default', 'id', 'maximum', 'minimum', 'name', 'network_type', 'physical_network', 'project_id', 'shared', 'used')
    data = (['100-103', '105'], _network_segment_range.default, _network_segment_range.id, _network_segment_range.maximum, _network_segment_range.minimum, _network_segment_range.name, _network_segment_range.network_type, _network_segment_range.physical_network, _network_segment_range.project_id, _network_segment_range.shared, {'3312e4ba67864b2eb53f3f41432f8efc': ['104', '106']})

    def setUp(self):
        super(TestShowNetworkSegmentRange, self).setUp()
        self.network_client.find_network_segment_range = mock.Mock(return_value=self._network_segment_range)
        self.cmd = network_segment_range.ShowNetworkSegmentRange(self.app, self.namespace)

    def test_show_no_options(self):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_show_all_options(self):
        arglist = [self._network_segment_range.id]
        verifylist = [('network_segment_range', self._network_segment_range.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_network_segment_range.assert_called_once_with(self._network_segment_range.id, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)