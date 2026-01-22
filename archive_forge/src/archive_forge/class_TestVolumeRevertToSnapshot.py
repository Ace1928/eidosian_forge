import copy
from unittest import mock
from cinderclient import api_versions
from openstack.block_storage.v3 import block_storage_summary as _summary
from openstack.block_storage.v3 import snapshot as _snapshot
from openstack.block_storage.v3 import volume as _volume
from openstack.test import fakes as sdk_fakes
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes
from openstackclient.volume.v3 import volume
class TestVolumeRevertToSnapshot(BaseVolumeTest):

    def setUp(self):
        super().setUp()
        self.volume = sdk_fakes.generate_fake_resource(_volume.Volume)
        self.snapshot = sdk_fakes.generate_fake_resource(_snapshot.Snapshot, volume_id=self.volume.id)
        self.volume_sdk_client.find_volume.return_value = self.volume
        self.volume_sdk_client.find_snapshot.return_value = self.snapshot
        self.cmd = volume.VolumeRevertToSnapshot(self.app, None)

    def test_volume_revert_to_snapshot_pre_340(self):
        arglist = [self.snapshot.id]
        verifylist = [('snapshot', self.snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.40 or greater is required', str(exc))

    def test_volume_revert_to_snapshot(self):
        self._set_mock_microversion('3.40')
        arglist = [self.snapshot.id]
        verifylist = [('snapshot', self.snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.volume_sdk_client.revert_volume_to_snapshot.assert_called_once_with(self.volume, self.snapshot)
        self.volume_sdk_client.find_volume.assert_called_with(self.volume.id, ignore_missing=False)
        self.volume_sdk_client.find_snapshot.assert_called_with(self.snapshot.id, ignore_missing=False)