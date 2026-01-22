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
def _wait_for_cluster_group_migration(self, event_listener, group_name, group_handle, expected_state, timeout=None):
    time_start = time.time()
    time_left = timeout if timeout else 'undefined'
    group_state_info = self._get_cluster_group_state(group_handle)
    group_state = group_state_info['state']
    group_status_info = group_state_info['status_info']
    migration_pending = self._is_migration_pending(group_state, group_status_info, expected_state)
    if not migration_pending:
        return
    while not timeout or time_left > 0:
        time_elapsed = time.time() - time_start
        time_left = timeout - time_elapsed if timeout else 'undefined'
        LOG.debug("Waiting for cluster group '%(group_name)s' migration to finish. Time left: %(time_left)s.", dict(group_name=group_name, time_left=time_left))
        try:
            event = event_listener.get(time_left if timeout else None)
        except queue.Empty:
            break
        group_state = event.get('state', group_state)
        group_status_info = event.get('status_info', group_status_info)
        migration_pending = self._is_migration_pending(group_state, group_status_info, expected_state)
        if not migration_pending:
            return
    LOG.error('Cluster group migration timed out.')
    raise exceptions.ClusterGroupMigrationTimeOut(group_name=group_name, time_elapsed=time.time() - time_start)