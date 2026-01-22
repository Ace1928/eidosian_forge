import platform
from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import migrationutils
from os_win.utils.compute import vmutils
def _get_vhd_setting_data(self, vm):
    new_resource_setting_data = []
    sasds = _wqlutils.get_element_associated_class(self._compat_conn, self._STORAGE_ALLOC_SETTING_DATA_CLASS, element_uuid=vm.Name)
    for sasd in sasds:
        if sasd.ResourceType == 31 and sasd.ResourceSubType == 'Microsoft:Hyper-V:Virtual Hard Disk':
            new_resource_setting_data.append(sasd.GetText_(1))
    return new_resource_setting_data