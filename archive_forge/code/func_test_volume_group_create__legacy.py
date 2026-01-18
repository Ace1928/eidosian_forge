from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
def test_volume_group_create__legacy(self):
    self.volume_client.api_version = api_versions.APIVersion('3.13')
    arglist = [self.fake_volume_group_type.id, self.fake_volume_type.id]
    verifylist = [('volume_group_type_legacy', self.fake_volume_group_type.id), ('volume_types_legacy', [self.fake_volume_type.id]), ('name', None), ('description', None), ('availability_zone', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch.object(self.cmd.log, 'warning') as mock_warning:
        columns, data = self.cmd.take_action(parsed_args)
    self.volume_group_types_mock.get.assert_called_once_with(self.fake_volume_group_type.id)
    self.volume_types_mock.get.assert_called_once_with(self.fake_volume_type.id)
    self.volume_groups_mock.create.assert_called_once_with(self.fake_volume_group_type.id, self.fake_volume_type.id, None, None, availability_zone=None)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)
    mock_warning.assert_called_once()
    self.assertIn('Passing volume group type and volume types as positional ', str(mock_warning.call_args[0][0]))