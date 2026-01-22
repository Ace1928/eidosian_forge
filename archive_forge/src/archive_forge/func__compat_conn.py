import importlib
import sys
import threading
import time
from oslo_log import log as logging
from oslo_utils import reflection
@property
def _compat_conn(self):
    if not self._compat_conn_attr:
        if not BaseUtilsVirt._os_version:
            os_version = wmi.WMI().Win32_OperatingSystem()[0].Version
            BaseUtilsVirt._os_version = list(map(int, os_version.split('.')))
        if BaseUtilsVirt._os_version >= [6, 3]:
            self._compat_conn_attr = self._conn
        else:
            self._compat_conn_attr = self._get_wmi_compat_conn(moniker=self._wmi_namespace % self._host)
    return self._compat_conn_attr