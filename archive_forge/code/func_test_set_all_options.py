from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment_range
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_all_options(self):
    arglist = ['--name', 'new name', '--minimum', str(self.minimum_updated), '--maximum', str(self.maximum_updated), self._network_segment_range.id]
    verifylist = [('name', 'new name'), ('minimum', self.minimum_updated), ('maximum', self.maximum_updated), ('network_segment_range', self._network_segment_range.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.network_client.update_network_segment_range = mock.Mock(return_value=self._network_segment_range_updated)
    result = self.cmd.take_action(parsed_args)
    attrs = {'name': 'new name', 'minimum': self.minimum_updated, 'maximum': self.maximum_updated}
    self.network_client.update_network_segment_range.assert_called_once_with(self._network_segment_range, **attrs)
    self.assertIsNone(result)