from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
@staticmethod
def _filter_metrics(all_metrics, metrics_def):
    return [v for v in all_metrics if v.MetricDefinitionId == metrics_def.Id]