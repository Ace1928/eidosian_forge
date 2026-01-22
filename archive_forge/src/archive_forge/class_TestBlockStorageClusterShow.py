from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_cluster
class TestBlockStorageClusterShow(TestBlockStorageCluster):
    cluster = volume_fakes.create_one_cluster()
    columns = ('Name', 'Binary', 'State', 'Status', 'Disabled Reason', 'Hosts', 'Down Hosts', 'Last Heartbeat', 'Created At', 'Updated At', 'Replication Status', 'Frozen', 'Active Backend ID')
    data = (cluster.name, cluster.binary, cluster.state, cluster.status, cluster.disabled_reason, cluster.num_hosts, cluster.num_down_hosts, cluster.last_heartbeat, cluster.created_at, cluster.updated_at, cluster.replication_status, cluster.frozen, cluster.active_backend_id)

    def setUp(self):
        super().setUp()
        self.cluster_mock.show.return_value = self.cluster
        self.cmd = block_storage_cluster.ShowBlockStorageCluster(self.app, None)

    def test_cluster_show(self):
        self.volume_client.api_version = api_versions.APIVersion('3.7')
        arglist = ['--binary', self.cluster.binary, self.cluster.name]
        verifylist = [('cluster', self.cluster.name), ('binary', self.cluster.binary)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))
        self.cluster_mock.show.assert_called_once_with(self.cluster.name, binary=self.cluster.binary)

    def test_cluster_show_pre_v37(self):
        self.volume_client.api_version = api_versions.APIVersion('3.6')
        arglist = ['--binary', self.cluster.binary, self.cluster.name]
        verifylist = [('cluster', self.cluster.name), ('binary', self.cluster.binary)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.7 or greater is required', str(exc))