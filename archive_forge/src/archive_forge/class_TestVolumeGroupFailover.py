from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group
class TestVolumeGroupFailover(TestVolumeGroup):
    fake_volume_group = volume_fakes.create_one_volume_group()

    def setUp(self):
        super().setUp()
        self.volume_groups_mock.get.return_value = self.fake_volume_group
        self.volume_groups_mock.failover_replication.return_value = None
        self.cmd = volume_group.FailoverVolumeGroup(self.app, None)

    def test_volume_group_failover(self):
        self.volume_client.api_version = api_versions.APIVersion('3.38')
        arglist = [self.fake_volume_group.id, '--allow-attached-volume', '--secondary-backend-id', 'foo']
        verifylist = [('group', self.fake_volume_group.id), ('allow_attached_volume', True), ('secondary_backend_id', 'foo')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volume_groups_mock.failover_replication.assert_called_once_with(self.fake_volume_group.id, allow_attached_volume=True, secondary_backend_id='foo')
        self.assertIsNone(result)

    def test_volume_group_failover_pre_v338(self):
        self.volume_client.api_version = api_versions.APIVersion('3.37')
        arglist = [self.fake_volume_group.id, '--allow-attached-volume', '--secondary-backend-id', 'foo']
        verifylist = [('group', self.fake_volume_group.id), ('allow_attached_volume', True), ('secondary_backend_id', 'foo')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.38 or greater is required', str(exc))