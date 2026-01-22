from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment_range
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetNetworkSegmentRange(TestNetworkSegmentRange):
    _network_segment_range = network_fakes.create_one_network_segment_range()
    minimum_updated = _network_segment_range.minimum - 5
    maximum_updated = _network_segment_range.maximum + 5
    available_updated = list(range(minimum_updated, 104)) + [105] + list(range(107, maximum_updated + 1))
    _network_segment_range_updated = network_fakes.create_one_network_segment_range(attrs={'minimum': minimum_updated, 'maximum': maximum_updated, 'used': {104: '3312e4ba67864b2eb53f3f41432f8efc', 106: '3312e4ba67864b2eb53f3f41432f8efc'}, 'available': available_updated})

    def setUp(self):
        super(TestSetNetworkSegmentRange, self).setUp()
        self.network_client.find_network_segment_range = mock.Mock(return_value=self._network_segment_range)
        self.cmd = network_segment_range.SetNetworkSegmentRange(self.app, self.namespace)

    def test_set_no_options(self):
        arglist = [self._network_segment_range.id]
        verifylist = [('network_segment_range', self._network_segment_range.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.network_client.update_network_segment_range = mock.Mock(return_value=self._network_segment_range)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_network_segment_range.assert_called_once_with(self._network_segment_range, **{})
        self.assertIsNone(result)

    def test_set_all_options(self):
        arglist = ['--name', 'new name', '--minimum', str(self.minimum_updated), '--maximum', str(self.maximum_updated), self._network_segment_range.id]
        verifylist = [('name', 'new name'), ('minimum', self.minimum_updated), ('maximum', self.maximum_updated), ('network_segment_range', self._network_segment_range.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.network_client.update_network_segment_range = mock.Mock(return_value=self._network_segment_range_updated)
        result = self.cmd.take_action(parsed_args)
        attrs = {'name': 'new name', 'minimum': self.minimum_updated, 'maximum': self.maximum_updated}
        self.network_client.update_network_segment_range.assert_called_once_with(self._network_segment_range, **attrs)
        self.assertIsNone(result)