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
def _get_disk_controller_type(self, controller_path):
    ctrl = self._get_wmi_obj(controller_path)
    res_sub_type = ctrl.ResourceSubType
    ctrl_type = self._disk_ctrl_type_mapping[res_sub_type]
    return ctrl_type