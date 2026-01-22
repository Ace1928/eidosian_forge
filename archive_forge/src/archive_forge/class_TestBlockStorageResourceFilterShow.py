from unittest import mock
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_resource_filter
class TestBlockStorageResourceFilterShow(TestBlockStorageResourceFilter):
    fake_resource_filter = volume_fakes.create_one_resource_filter()

    def setUp(self):
        super().setUp()
        self.volume_sdk_client.resource_filters.return_value = iter([self.fake_resource_filter])
        self.cmd = block_storage_resource_filter.ShowBlockStorageResourceFilter(self.app, None)

    def test_resource_filter_show(self):
        self._set_mock_microversion('3.33')
        arglist = [self.fake_resource_filter.resource]
        verifylist = [('resource', self.fake_resource_filter.resource)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        expected_columns = ('Resource', 'Filters')
        expected_data = (self.fake_resource_filter.resource, format_columns.ListColumn(self.fake_resource_filter.filters))
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, data)
        self.volume_sdk_client.resource_filters.assert_called_with(resource='volume')

    def test_resource_filter_show_pre_v333(self):
        self._set_mock_microversion('3.32')
        arglist = [self.fake_resource_filter.resource]
        verifylist = [('resource', self.fake_resource_filter.resource)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.33 or greater is required', str(exc))