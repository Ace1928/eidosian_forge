from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
def _get_vm_resources(self, vm_name, resource_class):
    setting_data = self._get_vm_setting_data(vm_name)
    return _wqlutils.get_element_associated_class(self._conn, resource_class, element_instance_id=setting_data.InstanceID)