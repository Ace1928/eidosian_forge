import importlib
import sys
import threading
import time
from oslo_log import log as logging
from oslo_utils import reflection
@property
def _vs_man_svc(self):
    if self._vs_man_svc_attr:
        return self._vs_man_svc_attr
    vs_man_svc = self._compat_conn.Msvm_VirtualSystemManagementService()[0]
    if BaseUtilsVirt._os_version >= [6, 3]:
        self._vs_man_svc_attr = vs_man_svc
    return vs_man_svc