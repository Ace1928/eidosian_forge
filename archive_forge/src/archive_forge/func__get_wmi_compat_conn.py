import importlib
import sys
import threading
import time
from oslo_log import log as logging
from oslo_utils import reflection
def _get_wmi_compat_conn(self, moniker, **kwargs):
    if not BaseUtilsVirt._old_wmi:
        old_wmi_path = '%s.py' % wmi.__path__[0]
        spec = importlib.util.spec_from_file_location('old_wmi', old_wmi_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        BaseUtilsVirt._old_wmi = module
    return BaseUtilsVirt._old_wmi.WMI(moniker=moniker, **kwargs)