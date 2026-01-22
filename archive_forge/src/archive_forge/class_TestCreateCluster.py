from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import cluster_templates as api_ct
from saharaclient.api import clusters as api_cl
from saharaclient.api import images as api_img
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import clusters as osc_cl
from saharaclient.tests.unit.osc.v1 import test_clusters as tc_v1
class TestCreateCluster(TestClusters):

    def setUp(self):
        super(TestCreateCluster, self).setUp()
        self.cl_mock.create.return_value = api_cl.Cluster(None, CLUSTER_INFO)
        self.cl_mock.find_unique.return_value = api_cl.Cluster(None, CLUSTER_INFO)
        self.ct_mock.find_unique.return_value = api_ct.ClusterTemplate(None, CT_INFO)
        self.img_mock.find_unique.return_value = api_img.Image(None, {'id': 'img_id'})
        self.net_mock = self.app.client_manager.network
        self.net_mock.find_network.return_value = mock.Mock(id='net_id')
        self.net_mock.reset_mock()
        self.cmd = osc_cl.CreateCluster(self.app, None)

    def test_cluster_create_minimum_options(self):
        arglist = ['--name', 'fake', '--cluster-template', 'template', '--image', 'ubuntu']
        verifylist = [('name', 'fake'), ('cluster_template', 'template'), ('image', 'ubuntu')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.cl_mock.create.assert_called_once_with(cluster_template_id='ct_id', count=None, default_image_id='img_id', description=None, plugin_version='0.1', is_protected=False, is_public=False, is_transient=False, name='fake', net_id=None, plugin_name='fake', user_keypair_id=None)

    def test_cluster_create_all_options(self):
        arglist = ['--name', 'fake', '--cluster-template', 'template', '--image', 'ubuntu', '--user-keypair', 'test', '--neutron-network', 'net', '--description', 'descr', '--transient', '--public', '--protected']
        verifylist = [('name', 'fake'), ('cluster_template', 'template'), ('image', 'ubuntu'), ('user_keypair', 'test'), ('neutron_network', 'net'), ('description', 'descr'), ('transient', True), ('public', True), ('protected', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.cl_mock.create.assert_called_once_with(cluster_template_id='ct_id', count=None, default_image_id='img_id', description='descr', plugin_version='0.1', is_protected=True, is_public=True, is_transient=True, name='fake', net_id='net_id', plugin_name='fake', user_keypair_id='test')
        expected_columns = ('Anti affinity', 'Cluster template id', 'Description', 'Id', 'Image', 'Is protected', 'Is public', 'Name', 'Neutron management network', 'Node groups', 'Plugin name', 'Plugin version', 'Status', 'Use autoconfig', 'User keypair id')
        self.assertEqual(expected_columns, columns)
        expected_data = ('', 'ct_id', 'Cluster template for tests', 'cluster_id', 'img_id', False, False, 'fake', 'net_id', 'fakeng:2', 'fake', '0.1', 'Active', True, 'test')
        self.assertEqual(expected_data, data)

    def test_cluster_create_with_count(self):
        clusters_mock = mock.Mock()
        clusters_mock.to_dict.return_value = {'clusters': [{'cluster': {'id': 'cluster1_id'}}, {'cluster': {'id': 'cluster2_id'}}]}
        self.cl_mock.create.return_value = clusters_mock
        arglist = ['--name', 'fake', '--cluster-template', 'template', '--image', 'ubuntu', '--count', '2']
        verifylist = [('name', 'fake'), ('cluster_template', 'template'), ('image', 'ubuntu'), ('count', 2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.cl_mock.create.assert_called_once_with(cluster_template_id='ct_id', count=2, default_image_id='img_id', description=None, plugin_version='0.1', is_protected=False, is_public=False, is_transient=False, name='fake', net_id=None, plugin_name='fake', user_keypair_id=None)
        expected_columns = ('fake',)
        self.assertEqual(expected_columns, columns)
        expected_data = ('cluster_id',)
        self.assertEqual(expected_data, data)