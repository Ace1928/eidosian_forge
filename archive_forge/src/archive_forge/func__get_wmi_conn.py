import importlib
import sys
import threading
import time
from oslo_log import log as logging
from oslo_utils import reflection
def _get_wmi_conn(self, moniker, **kwargs):
    if sys.platform != 'win32':
        return None
    if kwargs:
        return self._get_wmi_obj(moniker, **kwargs)
    if moniker in self._WMI_CONS:
        return self._WMI_CONS[moniker]
    wmi_conn = self._get_wmi_obj(moniker)
    self._WMI_CONS[moniker] = wmi_conn
    return wmi_conn