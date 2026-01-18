from unittest import mock
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_resource_filter
def test_resource_filter_show_pre_v333(self):
    self._set_mock_microversion('3.32')
    arglist = [self.fake_resource_filter.resource]
    verifylist = [('resource', self.fake_resource_filter.resource)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.33 or greater is required', str(exc))