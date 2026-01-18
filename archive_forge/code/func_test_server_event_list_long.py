from unittest import mock
import iso8601
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import server_event
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
def test_server_event_list_long(self):
    arglist = ['--long', self.fake_server.name]
    verifylist = [('server', self.fake_server.name), ('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.find_server.assert_called_with(self.fake_server.name, ignore_missing=False)
    self.compute_sdk_client.server_actions.assert_called_with(self.fake_server.id)
    self.assertEqual(self.long_columns, columns)
    self.assertEqual(self.long_data, tuple(data))