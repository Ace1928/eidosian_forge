from unittest import mock
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group_type
def test_volume_group_type_delete(self):
    self.volume_client.api_version = api_versions.APIVersion('3.11')
    arglist = [self.fake_volume_group_type.id]
    verifylist = [('group_type', self.fake_volume_group_type.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.volume_group_types_mock.delete.assert_called_once_with(self.fake_volume_group_type.id)
    self.assertIsNone(result)