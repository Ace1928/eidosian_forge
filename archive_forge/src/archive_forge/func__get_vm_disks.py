import functools
import time
import uuid
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import uuidutils
from six.moves import range  # noqa
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
from os_win.utils import pathutils
def _get_vm_disks(self, vmsettings):
    rasds = _wqlutils.get_element_associated_class(self._compat_conn, self._STORAGE_ALLOC_SETTING_DATA_CLASS, element_instance_id=vmsettings.InstanceID)
    disk_resources = [r for r in rasds if r.ResourceSubType in [self._HARD_DISK_RES_SUB_TYPE, self._DVD_DISK_RES_SUB_TYPE]]
    if self._RESOURCE_ALLOC_SETTING_DATA_CLASS != self._STORAGE_ALLOC_SETTING_DATA_CLASS:
        rasds = _wqlutils.get_element_associated_class(self._compat_conn, self._RESOURCE_ALLOC_SETTING_DATA_CLASS, element_instance_id=vmsettings.InstanceID)
    volume_resources = [r for r in rasds if r.ResourceSubType == self._PHYS_DISK_RES_SUB_TYPE]
    return (disk_resources, volume_resources)