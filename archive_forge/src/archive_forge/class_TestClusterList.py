import copy
import os
import sys
import tempfile
from unittest import mock
from contextlib import contextmanager
from unittest.mock import call
from magnumclient import exceptions
from magnumclient.osc.v1 import clusters as osc_clusters
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestClusterList(TestCluster):
    attr = dict()
    attr['name'] = 'fake-cluster-1'
    _cluster = magnum_fakes.FakeCluster.create_one_cluster(attr)
    columns = ['uuid', 'name', 'keypair', 'node_count', 'master_count', 'status', 'health_status']
    datalist = ((_cluster.uuid, _cluster.name, _cluster.keypair, _cluster.node_count, _cluster.master_count, _cluster.status, _cluster.health_status),)

    def setUp(self):
        super(TestClusterList, self).setUp()
        self.clusters_mock.list = mock.Mock()
        self.clusters_mock.list.return_value = [self._cluster]
        self.cmd = osc_clusters.ListCluster(self.app, None)

    def test_cluster_list_no_options(self):
        arglist = []
        verifylist = [('limit', None), ('sort_key', None), ('sort_dir', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.clusters_mock.list.assert_called_with(limit=None, sort_dir=None, sort_key=None)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_cluster_list_options(self):
        arglist = ['--limit', '1', '--sort-key', 'key', '--sort-dir', 'asc']
        verifylist = [('limit', 1), ('sort_key', 'key'), ('sort_dir', 'asc')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.clusters_mock.list.assert_called_with(limit=1, sort_dir='asc', sort_key='key')

    def test_cluster_list_bad_sort_dir_fail(self):
        arglist = ['--sort-dir', 'foo']
        verifylist = [('limit', None), ('sort_key', None), ('sort_dir', 'foo'), ('fields', None)]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)