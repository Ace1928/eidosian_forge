from unittest import mock
from magnumclient.tests.v1 import shell_test_base
@mock.patch('magnumclient.v1.quotas.QuotasManager.list')
def test_quotas_list_success(self, mock_list):
    self._test_arg_success('quotas-list')
    expected_args = self._get_expected_args_list()
    mock_list.assert_called_once_with(**expected_args)