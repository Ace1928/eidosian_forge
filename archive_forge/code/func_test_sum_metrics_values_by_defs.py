from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
def test_sum_metrics_values_by_defs(self):
    mock_metric = mock.MagicMock(MetricDefinitionId=mock.sentinel.def_id, MetricValue='100')
    mock_metric_useless = mock.MagicMock(MetricValue='200')
    mock_metric_def = mock.MagicMock(Id=mock.sentinel.def_id)
    result = self.utils._sum_metrics_values_by_defs([mock_metric, mock_metric_useless], [None, mock_metric_def])
    self.assertEqual([0, 100], result)