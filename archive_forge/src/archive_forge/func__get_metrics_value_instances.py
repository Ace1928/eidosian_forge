from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
def _get_metrics_value_instances(self, elements, result_class):
    instances = []
    for el in elements:
        el_metric = [x.Dependent for x in self._conn.Msvm_MetricForME(Antecedent=el.path_())]
        el_metric = [x for x in el_metric if x.path().Class == result_class]
        if el_metric:
            instances.append(el_metric[0])
    return instances