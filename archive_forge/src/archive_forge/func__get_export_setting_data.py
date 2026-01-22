from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
def _get_export_setting_data(self, vm_name):
    vm = self._vmutils._lookup_vm(vm_name)
    export_sd = self._compat_conn.Msvm_VirtualSystemExportSettingData(InstanceID=vm.InstanceID)
    return export_sd[0]