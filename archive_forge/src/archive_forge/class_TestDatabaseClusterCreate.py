from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_clusters
from troveclient.tests.osc.v1 import fakes
class TestDatabaseClusterCreate(TestClusters):
    values = ('2015-05-02T10:37:04', 'vertica', '7.1', 'cls-1234', 2, 'test-clstr', 'No tasks for the cluster.', 'NONE', '2015-05-02T11:06:19')
    columns = ('created', 'datastore', 'datastore_version', 'id', 'instance_count', 'name', 'task_description', 'task_name', 'updated')

    def setUp(self):
        super(TestDatabaseClusterCreate, self).setUp()
        self.cmd = database_clusters.CreateDatabaseCluster(self.app, None)
        self.data = self.fake_clusters.get_clusters_cls_1234()
        self.cluster_client.create.return_value = self.data

    @mock.patch.object(database_clusters, '_parse_instance_options')
    def test_cluster_create(self, mock_parse_instance_opts):
        instances = ['flavor="02",volume=2', 'flavor="03",volume=3']
        parsed_instances = [{'flavor': '02', 'volume': 2}, {'flavor': '03', 'volume': 3}]
        extended_properties = 'foo_properties=foo_value'
        parsed_extended_properties = {'foo_properties': 'foo_value'}
        mock_parse_instance_opts.return_value = parsed_instances
        args = ['test-name', 'vertica', '7.1', '--instance', instances[0], '--instance', instances[1], '--extended-properties', extended_properties, '--configuration', 'config01']
        verifylist = [('name', 'test-name'), ('datastore', 'vertica'), ('datastore_version', '7.1'), ('instances', instances), ('extended_properties', extended_properties), ('configuration', 'config01')]
        parsed_args = self.check_parser(self.cmd, args, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.cluster_client.create.assert_called_with(parsed_args.name, parsed_args.datastore, parsed_args.datastore_version, instances=parsed_instances, locality=parsed_args.locality, extended_properties=parsed_extended_properties, configuration=parsed_args.configuration)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)