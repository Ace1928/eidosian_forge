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
def update_vm_disk_path(self, disk_path, new_disk_path, is_physical=True):
    disk_resource = self._get_mounted_disk_resource_from_path(disk_path=disk_path, is_physical=is_physical)
    disk_resource.HostResource = [new_disk_path]
    self._jobutils.modify_virt_resource(disk_resource)