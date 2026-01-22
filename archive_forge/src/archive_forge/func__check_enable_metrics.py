from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
def _check_enable_metrics(self, metrics=None, definition=None):
    mock_element = mock.MagicMock()
    self.utils._metrics_svc.ControlMetrics.return_value = [0]
    self.utils._enable_metrics(mock_element, metrics)
    self.utils._metrics_svc.ControlMetrics.assert_called_once_with(Subject=mock_element.path_.return_value, Definition=definition, MetricCollectionEnabled=self.utils._METRICS_ENABLED)