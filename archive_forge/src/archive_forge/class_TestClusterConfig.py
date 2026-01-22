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
class TestClusterConfig(TestCluster):

    def setUp(self):
        super(TestClusterConfig, self).setUp()
        attr = dict()
        attr['name'] = 'fake-cluster-1'
        attr['status'] = 'CREATE_COMPLETE'
        self._cluster = magnum_fakes.FakeCluster.create_one_cluster(attr)
        self.clusters_mock.get = mock.Mock()
        self.clusters_mock.get.return_value = self._cluster
        cert = magnum_fakes.FakeCert(pem='foo bar')
        self.certificates_mock.create = mock.Mock()
        self.certificates_mock.create.return_value = cert
        self.certificates_mock.get = mock.Mock()
        self.certificates_mock.get.return_value = cert
        attr = dict()
        attr['name'] = 'fake-ct'
        self._cluster_template = magnum_fakes.FakeClusterTemplate.create_one_cluster_template(attr)
        self.cluster_templates_mock = self.app.client_manager.container_infra.cluster_templates
        self.cluster_templates_mock.get = mock.Mock()
        self.cluster_templates_mock.get.return_value = self._cluster_template
        self.cmd = osc_clusters.ConfigCluster(self.app, None)

    def test_cluster_config_no_cluster_fail(self):
        arglist = []
        verifylist = []
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)

    @mock.patch.dict(os.environ, {'SHELL': '/bin/bash'})
    def test_cluster_config_custom_dir_with_config_only_works_if_force(self):
        tmp_dir = tempfile.mkdtemp()
        open(os.path.join(tmp_dir, 'config'), 'a').close()
        arglist = ['fake-cluster', '--dir', tmp_dir]
        verifylist = [('cluster', 'fake-cluster'), ('force', False), ('dir', tmp_dir)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.clusters_mock.get.assert_called_with('fake-cluster')
        arglist = ['fake-cluster', '--force', '--dir', tmp_dir]
        verifylist = [('cluster', 'fake-cluster'), ('force', True), ('dir', tmp_dir)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        expected_value = 'export KUBECONFIG={}/config\n\n'.format(tmp_dir)
        with capture(self.cmd.take_action, parsed_args) as output:
            self.assertEqual(expected_value, output)
        self.clusters_mock.get.assert_called_with('fake-cluster')

    @mock.patch.dict(os.environ, {'SHELL': '/bin/bash'})
    def test_cluster_config_with_custom_dir(self):
        tmp_dir = tempfile.mkdtemp()
        arglist = ['fake-cluster', '--dir', tmp_dir]
        verifylist = [('cluster', 'fake-cluster'), ('dir', tmp_dir)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        expected_value = 'export KUBECONFIG={}/config\n\n'.format(tmp_dir)
        with capture(self.cmd.take_action, parsed_args) as output:
            self.assertEqual(expected_value, output)
        self.clusters_mock.get.assert_called_with('fake-cluster')

    @mock.patch.dict(os.environ, {'SHELL': '/bin/bash'})
    def test_cluster_config_creates_config_in_cwd_if_not_dir_specified(self):
        tmp_dir = tempfile.mkdtemp()
        os.chdir(tmp_dir)
        arglist = ['fake-cluster']
        verifylist = [('cluster', 'fake-cluster')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        expected_value = 'export KUBECONFIG={}/config\n\n'.format(os.getcwd())
        with capture(self.cmd.take_action, parsed_args) as output:
            self.assertEqual(expected_value, output)
        self.clusters_mock.get.assert_called_with('fake-cluster')