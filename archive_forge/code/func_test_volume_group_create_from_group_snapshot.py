from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
def test_volume_group_create_from_group_snapshot(self):
    self.volume_client.api_version = api_versions.APIVersion('3.14')
    arglist = ['--group-snapshot', self.fake_volume_group_snapshot.id]
    verifylist = [('group_snapshot', self.fake_volume_group_snapshot.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_group_snapshots_mock.get.assert_called_once_with(self.fake_volume_group_snapshot.id)
    self.volume_groups_mock.get.assert_called_once_with(self.fake_volume_group.id)
    self.volume_groups_mock.create_from_src.assert_called_once_with(self.fake_volume_group_snapshot.id, None, None, None)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)