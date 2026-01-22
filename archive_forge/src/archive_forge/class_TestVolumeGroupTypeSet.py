from unittest import mock
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group_type
class TestVolumeGroupTypeSet(TestVolumeGroupType):
    fake_volume_group_type = volume_fakes.create_one_volume_group_type(methods={'get_keys': {'foo': 'bar'}, 'set_keys': None, 'unset_keys': None})
    columns = ('ID', 'Name', 'Description', 'Is Public', 'Properties')
    data = (fake_volume_group_type.id, fake_volume_group_type.name, fake_volume_group_type.description, fake_volume_group_type.is_public, format_columns.DictColumn(fake_volume_group_type.group_specs))

    def setUp(self):
        super().setUp()
        self.volume_group_types_mock.get.return_value = self.fake_volume_group_type
        self.volume_group_types_mock.update.return_value = self.fake_volume_group_type
        self.cmd = volume_group_type.SetVolumeGroupType(self.app, None)

    def test_volume_group_type_set(self):
        self.volume_client.api_version = api_versions.APIVersion('3.11')
        self.fake_volume_group_type.set_keys.return_value = None
        arglist = [self.fake_volume_group_type.id, '--name', 'foo', '--description', 'hello, world', '--public', '--property', 'fizz=buzz']
        verifylist = [('group_type', self.fake_volume_group_type.id), ('name', 'foo'), ('description', 'hello, world'), ('is_public', True), ('no_property', False), ('properties', {'fizz': 'buzz'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_group_types_mock.update.assert_called_once_with(self.fake_volume_group_type.id, name='foo', description='hello, world', is_public=True)
        self.fake_volume_group_type.set_keys.assert_called_once_with({'fizz': 'buzz'})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_volume_group_type_with_no_property_option(self):
        self.volume_client.api_version = api_versions.APIVersion('3.11')
        arglist = [self.fake_volume_group_type.id, '--no-property', '--property', 'fizz=buzz']
        verifylist = [('group_type', self.fake_volume_group_type.id), ('name', None), ('description', None), ('is_public', None), ('no_property', True), ('properties', {'fizz': 'buzz'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_group_types_mock.get.assert_called_once_with(self.fake_volume_group_type.id)
        self.fake_volume_group_type.get_keys.assert_called_once_with()
        self.fake_volume_group_type.unset_keys.assert_called_once_with({'foo': 'bar'}.keys())
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_volume_group_type_set_pre_v311(self):
        self.volume_client.api_version = api_versions.APIVersion('3.10')
        arglist = [self.fake_volume_group_type.id, '--name', 'foo', '--description', 'hello, world']
        verifylist = [('group_type', self.fake_volume_group_type.id), ('name', 'foo'), ('description', 'hello, world'), ('is_public', None), ('no_property', False), ('properties', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.11 or greater is required', str(exc))