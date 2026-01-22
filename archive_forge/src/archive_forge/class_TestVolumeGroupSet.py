from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
class TestVolumeGroupSet(TestVolumeGroup):
    fake_volume_group = volume_fakes.create_one_volume_group()
    columns = ('ID', 'Status', 'Name', 'Description', 'Group Type', 'Volume Types', 'Availability Zone', 'Created At', 'Volumes', 'Group Snapshot ID', 'Source Group ID')
    data = (fake_volume_group.id, fake_volume_group.status, fake_volume_group.name, fake_volume_group.description, fake_volume_group.group_type, fake_volume_group.volume_types, fake_volume_group.availability_zone, fake_volume_group.created_at, fake_volume_group.volumes, fake_volume_group.group_snapshot_id, fake_volume_group.source_group_id)

    def setUp(self):
        super().setUp()
        self.volume_groups_mock.get.return_value = self.fake_volume_group
        self.volume_groups_mock.update.return_value = self.fake_volume_group
        self.cmd = volume_group.SetVolumeGroup(self.app, None)

    def test_volume_group_set(self):
        self.volume_client.api_version = api_versions.APIVersion('3.13')
        arglist = [self.fake_volume_group.id, '--name', 'foo', '--description', 'hello, world']
        verifylist = [('group', self.fake_volume_group.id), ('name', 'foo'), ('description', 'hello, world')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_groups_mock.update.assert_called_once_with(self.fake_volume_group.id, name='foo', description='hello, world')
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_volume_group_with_enable_replication_option(self):
        self.volume_client.api_version = api_versions.APIVersion('3.38')
        arglist = [self.fake_volume_group.id, '--enable-replication']
        verifylist = [('group', self.fake_volume_group.id), ('enable_replication', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_groups_mock.enable_replication.assert_called_once_with(self.fake_volume_group.id)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_volume_group_set_pre_v313(self):
        self.volume_client.api_version = api_versions.APIVersion('3.12')
        arglist = [self.fake_volume_group.id, '--name', 'foo', '--description', 'hello, world']
        verifylist = [('group', self.fake_volume_group.id), ('name', 'foo'), ('description', 'hello, world')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.13 or greater is required', str(exc))

    def test_volume_group_with_enable_replication_option_pre_v338(self):
        self.volume_client.api_version = api_versions.APIVersion('3.37')
        arglist = [self.fake_volume_group.id, '--enable-replication']
        verifylist = [('group', self.fake_volume_group.id), ('enable_replication', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.38 or greater is required', str(exc))