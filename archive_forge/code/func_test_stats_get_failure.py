from unittest import mock
from magnumclient.tests.v1 import shell_test_base
from magnumclient.v1.stats import Stats
@mock.patch('magnumclient.v1.stats.StatsManager.list')
def test_stats_get_failure(self, mock_list):
    self._test_arg_failure('stats-list --wrong', self._unrecognized_arg_error)
    mock_list.assert_not_called()