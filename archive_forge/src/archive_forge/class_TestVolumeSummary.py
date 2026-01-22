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
class TestVolumeSummary(BaseVolumeTest):
    columns = ['Total Count', 'Total Size']

    def setUp(self):
        super().setUp()
        self.volume_a = sdk_fakes.generate_fake_resource(_volume.Volume)
        self.volume_b = sdk_fakes.generate_fake_resource(_volume.Volume)
        self.summary = sdk_fakes.generate_fake_resource(_summary.BlockStorageSummary, total_count=2, total_size=self.volume_a.size + self.volume_b.size)
        self.volume_sdk_client.summary.return_value = self.summary
        self.cmd = volume.VolumeSummary(self.app, None)

    def test_volume_summary(self):
        self._set_mock_microversion('3.12')
        arglist = ['--all-projects']
        verifylist = [('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_sdk_client.summary.assert_called_once_with(True)
        self.assertEqual(self.columns, columns)
        datalist = (2, self.volume_a.size + self.volume_b.size)
        self.assertCountEqual(datalist, tuple(data))

    def test_volume_summary_pre_312(self):
        arglist = ['--all-projects']
        verifylist = [('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.12 or greater is required', str(exc))

    def test_volume_summary_with_metadata(self):
        self._set_mock_microversion('3.36')
        metadata = {**self.volume_a.metadata, **self.volume_b.metadata}
        self.summary = sdk_fakes.generate_fake_resource(_summary.BlockStorageSummary, total_count=2, total_size=self.volume_a.size + self.volume_b.size, metadata=metadata)
        self.volume_sdk_client.summary.return_value = self.summary
        new_cols = copy.deepcopy(self.columns)
        new_cols.extend(['Metadata'])
        arglist = ['--all-projects']
        verifylist = [('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_sdk_client.summary.assert_called_once_with(True)
        self.assertEqual(new_cols, columns)
        datalist = (2, self.volume_a.size + self.volume_b.size, format_columns.DictColumn(metadata))
        self.assertCountEqual(datalist, tuple(data))