from unittest import mock
from magnumclient.tests.v1 import shell_test_base
@mock.patch('magnumclient.v1.quotas.QuotasManager.get')
def test_quotas_show_failure_no_arg(self, mock_show):
    self._test_arg_failure('quotas-show', self._mandatory_arg_error)
    mock_show.assert_not_called()