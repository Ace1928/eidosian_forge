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
class TestClusterShow(TestCluster):

    def setUp(self):
        super(TestClusterShow, self).setUp()
        attr = dict()
        attr['name'] = 'fake-cluster-1'
        self._cluster = magnum_fakes.FakeCluster.create_one_cluster(attr)
        self.clusters_mock.get = mock.Mock()
        self.clusters_mock.get.return_value = self._cluster
        self.cmd = osc_clusters.ShowCluster(self.app, None)
        self.data = tuple(map(lambda x: getattr(self._cluster, x), osc_clusters.CLUSTER_ATTRIBUTES))

    def test_cluster_show_pass(self):
        arglist = ['fake-cluster']
        verifylist = [('cluster', 'fake-cluster')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.clusters_mock.get.assert_called_with('fake-cluster')
        self.assertEqual(osc_clusters.CLUSTER_ATTRIBUTES, columns)
        self.assertEqual(self.data, data)

    def test_cluster_show_no_cluster_fail(self):
        arglist = []
        verifylist = []
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)