import importlib
import sys
import threading
import time
from oslo_log import log as logging
from oslo_utils import reflection
def _get_wmi_obj(self, moniker, compatibility_mode=False, **kwargs):
    if not BaseUtilsVirt._os_version:
        os_version = wmi.WMI().Win32_OperatingSystem()[0].Version
        BaseUtilsVirt._os_version = list(map(int, os_version.split('.')))
    if not compatibility_mode or BaseUtilsVirt._os_version >= [6, 3]:
        return wmi.WMI(moniker=moniker, **kwargs)
    return self._get_wmi_compat_conn(moniker=moniker, **kwargs)