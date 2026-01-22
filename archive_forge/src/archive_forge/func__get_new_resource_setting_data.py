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
def _get_new_resource_setting_data(self, resource_sub_type, class_name=None):
    if class_name is None:
        class_name = self._RESOURCE_ALLOC_SETTING_DATA_CLASS
    obj = self._compat_conn.query("SELECT * FROM %(class_name)s WHERE ResourceSubType = '%(res_sub_type)s' AND InstanceID LIKE '%%\\Default'" % {'class_name': class_name, 'res_sub_type': resource_sub_type})[0]
    return obj