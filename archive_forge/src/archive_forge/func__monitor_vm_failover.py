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
def _monitor_vm_failover(self, watcher, callback, event_timeout_ms=_WMI_EVENT_TIMEOUT_MS):
    """Creates a monitor to check for new WMI MSCluster_Resource

        events.

        This method will poll the last _WMI_EVENT_CHECK_INTERVAL + 1
        seconds for new events and listens for _WMI_EVENT_TIMEOUT_MS
        milliseconds, since listening is a thread blocking action.

        Any event object caught will then be processed.
        """
    vm_name = None
    new_host = None
    try:
        if patcher.is_monkey_patched('thread'):
            wmi_object = tpool.execute(watcher, event_timeout_ms)
        else:
            wmi_object = watcher(event_timeout_ms)
        old_host = wmi_object.previous.OwnerNode
        new_host = wmi_object.OwnerNode
        match = self._instance_name_regex.search(wmi_object.Name)
        if match:
            vm_name = match.group(1)
        if vm_name:
            try:
                callback(vm_name, old_host, new_host)
            except Exception:
                LOG.exception('Exception during failover callback.')
    except exceptions.x_wmi_timed_out:
        pass