from unittest import mock
from magnumclient.common import cliutils
from magnumclient import exceptions
from magnumclient.tests.v1 import shell_test_base
from magnumclient.tests.v1 import test_clustertemplates_shell
from magnumclient.v1.clusters import Cluster
@mock.patch('os.path.exists')
@mock.patch('magnumclient.v1.certificates.CertificateManager.create')
@mock.patch('magnumclient.v1.certificates.CertificateManager.get')
@mock.patch('magnumclient.v1.cluster_templates.ClusterTemplateManager.get')
@mock.patch('magnumclient.v1.clusters.ClusterManager.get')
def _test_cluster_config_success(self, mock_cluster, mock_ct, mock_cert_get, mock_cert_create, mock_exists, coe, shell, tls_disable):
    cert = FakeCert(pem='foo bar')
    mock_exists.return_value = False
    mock_cluster.return_value = FakeCluster(status='CREATE_COMPLETE', info={'name': 'Kluster', 'api_address': '10.0.0.1'}, cluster_template_id='fake_ct', uuid='fake_cluster')
    mock_cert_get.return_value = cert
    mock_cert_create.return_value = cert
    mock_ct.return_value = test_clustertemplates_shell.FakeClusterTemplate(coe=coe, name='fake_ct', tls_disabled=tls_disable)
    with mock.patch.dict('os.environ', {'SHELL': shell}):
        self._test_arg_success('cluster-config test_cluster')
    self.assertTrue(mock_exists.called)
    mock_cluster.assert_called_once_with('test_cluster')
    mock_ct.assert_called_once_with('fake_ct')
    if not tls_disable:
        mock_cert_create.assert_called_once_with(cluster_uuid='fake_cluster', csr=mock.ANY)
        mock_cert_get.assert_called_once_with(cluster_uuid='fake_cluster')