from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
def _cache_metrics_defs(self):
    for metrics_def in self._conn.CIM_BaseMetricDefinition():
        self._metrics_defs_obj[metrics_def.ElementName] = metrics_def