from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
def _get_metrics_values(self, element, metrics_defs):
    element_metrics = [x.Dependent for x in self._conn.Msvm_MetricForME(Antecedent=element.path_())]
    return self._sum_metrics_values_by_defs(element_metrics, metrics_defs)