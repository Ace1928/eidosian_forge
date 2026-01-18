from unittest import mock
from magnumclient.tests.v1 import shell_test_base
@mock.patch('magnumclient.v1.quotas.QuotasManager.create')
def test_quotas_create_failure_only_hard_limit(self, mock_create):
    self._test_arg_failure('quotas-create --hard-limit 10', self._mandatory_arg_error)
    mock_create.assert_not_called()