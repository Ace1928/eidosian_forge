from unittest import mock
from magnumclient.tests.v1 import shell_test_base
@mock.patch('magnumclient.v1.quotas.QuotasManager.create')
def test_quotas_create_success(self, mock_create):
    self._test_arg_success('quotas-create --project-id abc --resource Cluster --hard-limit 15')
    expected_args = self._get_expected_args_create('abc', 'Cluster', 15)
    mock_create.assert_called_with(**expected_args)