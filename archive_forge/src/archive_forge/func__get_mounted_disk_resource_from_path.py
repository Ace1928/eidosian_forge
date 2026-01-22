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
def _get_mounted_disk_resource_from_path(self, disk_path, is_physical, serial=None):
    if is_physical:
        class_name = self._RESOURCE_ALLOC_SETTING_DATA_CLASS
    else:
        class_name = self._STORAGE_ALLOC_SETTING_DATA_CLASS
    query = "SELECT * FROM %(class_name)s WHERE (ResourceSubType='%(res_sub_type)s' OR ResourceSubType='%(res_sub_type_virt)s' OR ResourceSubType='%(res_sub_type_dvd)s')" % {'class_name': class_name, 'res_sub_type': self._PHYS_DISK_RES_SUB_TYPE, 'res_sub_type_virt': self._HARD_DISK_RES_SUB_TYPE, 'res_sub_type_dvd': self._DVD_DISK_RES_SUB_TYPE}
    if serial:
        query += " AND ElementName='%s'" % serial
    disk_resources = self._compat_conn.query(query)
    for disk_resource in disk_resources:
        if serial:
            return disk_resource
        if disk_resource.HostResource:
            if disk_resource.HostResource[0].lower() == disk_path.lower():
                return disk_resource