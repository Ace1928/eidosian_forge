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
def _get_setting_data(self, class_name, element_name, create=True):
    element_name = element_name.replace("'", '"')
    q = self._compat_conn.query("SELECT * FROM %(class_name)s WHERE ElementName = '%(element_name)s'" % {'class_name': class_name, 'element_name': element_name})
    data = self._get_first_item(q)
    found = data is not None
    if not data and create:
        data = self._get_default_setting_data(class_name)
        data.ElementName = element_name
    return (data, found)