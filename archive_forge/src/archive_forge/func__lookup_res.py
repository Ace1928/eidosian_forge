import ctypes
import re
import sys
import threading
import time
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import excutils
from six.moves import queue
from os_win._i18n import _
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.compute import _clusapi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def _lookup_res(self, resource_source, res_name):
    res = resource_source(Name=res_name)
    n = len(res)
    if n == 0:
        return None
    elif n > 1:
        raise exceptions.HyperVClusterException(_('Duplicate resource name %s found.') % res_name)
    else:
        return res[0]