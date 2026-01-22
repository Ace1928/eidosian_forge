from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
def _get_planned_vm(self, vm_name, conn_v2=None, fail_if_not_found=False):
    if not conn_v2:
        conn_v2 = self._conn
    planned_vm = conn_v2.Msvm_PlannedComputerSystem(ElementName=vm_name)
    if planned_vm:
        return planned_vm[0]
    elif fail_if_not_found:
        raise exceptions.HyperVException(_('Cannot find planned VM with name: %s') % vm_name)
    return None