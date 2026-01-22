import functools
import re
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win._i18n import _
from os_win import conf
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
def _is_port_vm_started(self, port):
    vmsettings_instance_id = port.InstanceID.split('\\')[0]
    vmsettings = self._conn.Msvm_VirtualSystemSettingData(InstanceID=vmsettings_instance_id)
    ret_val, summary_info = self._vs_man_svc.GetSummaryInformation([self._VM_SUMMARY_ENABLED_STATE], [v.path_() for v in vmsettings])
    if ret_val or not summary_info:
        raise exceptions.HyperVException(_('Cannot get VM summary data for: %s') % port.ElementName)
    return summary_info[0].EnabledState == self._HYPERV_VM_STATE_ENABLED