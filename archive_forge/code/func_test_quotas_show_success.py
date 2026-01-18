from unittest import mock
from magnumclient.tests.v1 import shell_test_base
@mock.patch('magnumclient.v1.quotas.QuotasManager.get')
def test_quotas_show_success(self, mock_show):
    self._test_arg_success('quotas-show --project-id abc --resource Cluster')
    mock_show.assert_called_once_with('abc', 'Cluster')