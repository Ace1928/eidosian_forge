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
def _get_failover_watcher(self):
    raw_query = "SELECT * FROM __InstanceModificationEvent WITHIN %(wmi_check_interv)s WHERE TargetInstance ISA '%(cluster_res)s' AND TargetInstance.Type='%(cluster_res_type)s' AND TargetInstance.OwnerNode != PreviousInstance.OwnerNode" % {'wmi_check_interv': self._WMI_EVENT_CHECK_INTERVAL, 'cluster_res': self._MSCLUSTER_RES, 'cluster_res_type': self._VM_TYPE}
    return self._conn_cluster.watch_for(raw_wql=raw_query)