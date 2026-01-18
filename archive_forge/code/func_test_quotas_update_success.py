from unittest import mock
from magnumclient.tests.v1 import shell_test_base
@mock.patch('magnumclient.v1.quotas.QuotasManager.update')
def test_quotas_update_success(self, mock_update):
    self._test_arg_success('quotas-update --project-id abc --resource Cluster --hard-limit 20')
    patch = {'project_id': 'abc', 'resource': 'Cluster', 'hard_limit': 20}
    mock_update.assert_called_once_with('abc', 'Cluster', patch)