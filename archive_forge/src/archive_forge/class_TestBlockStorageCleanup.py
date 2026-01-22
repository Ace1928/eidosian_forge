import uuid
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_cleanup
class TestBlockStorageCleanup(TestBlockStorage):
    cleaning, unavailable = volume_fakes.create_cleanup_records()

    def setUp(self):
        super().setUp()
        self.worker_mock.clean.return_value = (self.cleaning, self.unavailable)
        self.cmd = block_storage_cleanup.BlockStorageCleanup(self.app, None)

    def test_cleanup(self):
        self.volume_client.api_version = api_versions.APIVersion('3.24')
        arglist = []
        verifylist = [('cluster', None), ('host', None), ('binary', None), ('is_up', None), ('disabled', None), ('resource_id', None), ('resource_type', None), ('service_id', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        expected_columns = ('ID', 'Cluster Name', 'Host', 'Binary', 'Status')
        cleaning_data = tuple(((obj.id, obj.cluster_name, obj.host, obj.binary, 'Cleaning') for obj in self.cleaning))
        unavailable_data = tuple(((obj.id, obj.cluster_name, obj.host, obj.binary, 'Unavailable') for obj in self.unavailable))
        expected_data = cleaning_data + unavailable_data
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, tuple(data))
        self.worker_mock.clean.assert_called_once_with()

    def test_block_storage_cleanup_pre_324(self):
        arglist = []
        verifylist = [('cluster', None), ('host', None), ('binary', None), ('is_up', None), ('disabled', None), ('resource_id', None), ('resource_type', None), ('service_id', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.24 or greater is required', str(exc))

    def test_cleanup_with_args(self):
        self.volume_client.api_version = api_versions.APIVersion('3.24')
        fake_cluster = 'fake-cluster'
        fake_host = 'fake-host'
        fake_binary = 'fake-service'
        fake_resource_id = str(uuid.uuid4())
        fake_resource_type = 'Volume'
        fake_service_id = 1
        arglist = ['--cluster', fake_cluster, '--host', fake_host, '--binary', fake_binary, '--down', '--enabled', '--resource-id', fake_resource_id, '--resource-type', fake_resource_type, '--service-id', str(fake_service_id)]
        verifylist = [('cluster', fake_cluster), ('host', fake_host), ('binary', fake_binary), ('is_up', False), ('disabled', False), ('resource_id', fake_resource_id), ('resource_type', fake_resource_type), ('service_id', fake_service_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        expected_columns = ('ID', 'Cluster Name', 'Host', 'Binary', 'Status')
        cleaning_data = tuple(((obj.id, obj.cluster_name, obj.host, obj.binary, 'Cleaning') for obj in self.cleaning))
        unavailable_data = tuple(((obj.id, obj.cluster_name, obj.host, obj.binary, 'Unavailable') for obj in self.unavailable))
        expected_data = cleaning_data + unavailable_data
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, tuple(data))
        self.worker_mock.clean.assert_called_once_with(cluster_name=fake_cluster, host=fake_host, binary=fake_binary, is_up=False, disabled=False, resource_id=fake_resource_id, resource_type=fake_resource_type, service_id=fake_service_id)