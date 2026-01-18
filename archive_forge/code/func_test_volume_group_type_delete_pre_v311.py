from unittest import mock
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group_type
def test_volume_group_type_delete_pre_v311(self):
    self.volume_client.api_version = api_versions.APIVersion('3.10')
    arglist = [self.fake_volume_group_type.id]
    verifylist = [('group_type', self.fake_volume_group_type.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-volume-api-version 3.11 or greater is required', str(exc))