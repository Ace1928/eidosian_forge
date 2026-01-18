from unittest import mock
import iso8601
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import server_event
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
@mock.patch.object(iso8601, 'parse_date', side_effect=iso8601.ParseError)
def test_server_event_list_with_changes_since_invalid(self, mock_parse_isotime):
    self._set_mock_microversion('2.58')
    arglist = ['--changes-since', 'Invalid time value', self.fake_server.name]
    verifylist = [('server', self.fake_server.name), ('changes_since', 'Invalid time value')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('Invalid changes-since value:', str(ex))
    mock_parse_isotime.assert_called_once_with('Invalid time value')