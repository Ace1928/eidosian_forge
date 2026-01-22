from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
class TestVolumeGroupList(TestVolumeGroup):
    fake_volume_groups = volume_fakes.create_volume_groups()
    columns = ('ID', 'Status', 'Name')
    data = [(fake_volume_group.id, fake_volume_group.status, fake_volume_group.name) for fake_volume_group in fake_volume_groups]

    def setUp(self):
        super().setUp()
        self.volume_groups_mock.list.return_value = self.fake_volume_groups
        self.cmd = volume_group.ListVolumeGroup(self.app, None)

    def test_volume_group_list(self):
        self.volume_client.api_version = api_versions.APIVersion('3.13')
        arglist = ['--all-projects']
        verifylist = [('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_groups_mock.list.assert_called_once_with(search_opts={'all_tenants': True})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(tuple(self.data), data)

    def test_volume_group_list_pre_v313(self):
        self.volume_client.api_version = api_versions.APIVersion('3.12')
        arglist = ['--all-projects']
        verifylist = [('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.13 or greater is required', str(exc))