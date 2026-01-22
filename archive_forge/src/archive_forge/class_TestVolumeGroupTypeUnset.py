from unittest import mock
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group_type
class TestVolumeGroupTypeUnset(TestVolumeGroupType):
    fake_volume_group_type = volume_fakes.create_one_volume_group_type(methods={'unset_keys': None})
    columns = ('ID', 'Name', 'Description', 'Is Public', 'Properties')
    data = (fake_volume_group_type.id, fake_volume_group_type.name, fake_volume_group_type.description, fake_volume_group_type.is_public, format_columns.DictColumn(fake_volume_group_type.group_specs))

    def setUp(self):
        super().setUp()
        self.volume_group_types_mock.get.return_value = self.fake_volume_group_type
        self.cmd = volume_group_type.UnsetVolumeGroupType(self.app, None)

    def test_volume_group_type_unset(self):
        self.volume_client.api_version = api_versions.APIVersion('3.11')
        arglist = [self.fake_volume_group_type.id, '--property', 'fizz']
        verifylist = [('group_type', self.fake_volume_group_type.id), ('properties', ['fizz'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_group_types_mock.get.assert_has_calls([mock.call(self.fake_volume_group_type.id), mock.call(self.fake_volume_group_type.id)])
        self.fake_volume_group_type.unset_keys.assert_called_once_with(['fizz'])
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_volume_group_type_unset_pre_v311(self):
        self.volume_client.api_version = api_versions.APIVersion('3.10')
        arglist = [self.fake_volume_group_type.id, '--property', 'fizz']
        verifylist = [('group_type', self.fake_volume_group_type.id), ('properties', ['fizz'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.11 or greater is required', str(exc))