from unittest import mock
import iso8601
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import server_event
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
def test_server_event_list_with_limit_pre_v258(self):
    self._set_mock_microversion('2.57')
    arglist = ['--limit', '1', self.fake_server.name]
    verifylist = [('limit', 1), ('server', self.fake_server.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-compute-api-version 2.58 or greater is required', str(ex))