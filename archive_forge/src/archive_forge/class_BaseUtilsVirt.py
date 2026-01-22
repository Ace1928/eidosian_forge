import importlib
import sys
import threading
import time
from oslo_log import log as logging
from oslo_utils import reflection
class BaseUtilsVirt(BaseUtils):
    _wmi_namespace = '//%s/root/virtualization/v2'
    _os_version = None
    _old_wmi = None

    def __init__(self, host='.'):
        self._vs_man_svc_attr = None
        self._host = host
        self._conn_attr = None
        self._compat_conn_attr = None

    @property
    def _conn(self):
        if not self._conn_attr:
            self._conn_attr = self._get_wmi_conn(self._wmi_namespace % self._host)
        return self._conn_attr

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

    @property
    def _vs_man_svc(self):
        if self._vs_man_svc_attr:
            return self._vs_man_svc_attr
        vs_man_svc = self._compat_conn.Msvm_VirtualSystemManagementService()[0]
        if BaseUtilsVirt._os_version >= [6, 3]:
            self._vs_man_svc_attr = vs_man_svc
        return vs_man_svc

    def _get_wmi_compat_conn(self, moniker, **kwargs):
        if not BaseUtilsVirt._old_wmi:
            old_wmi_path = '%s.py' % wmi.__path__[0]
            spec = importlib.util.spec_from_file_location('old_wmi', old_wmi_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            BaseUtilsVirt._old_wmi = module
        return BaseUtilsVirt._old_wmi.WMI(moniker=moniker, **kwargs)

    def _get_wmi_obj(self, moniker, compatibility_mode=False, **kwargs):
        if not BaseUtilsVirt._os_version:
            os_version = wmi.WMI().Win32_OperatingSystem()[0].Version
            BaseUtilsVirt._os_version = list(map(int, os_version.split('.')))
        if not compatibility_mode or BaseUtilsVirt._os_version >= [6, 3]:
            return wmi.WMI(moniker=moniker, **kwargs)
        return self._get_wmi_compat_conn(moniker=moniker, **kwargs)