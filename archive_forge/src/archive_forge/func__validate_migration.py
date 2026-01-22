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
def _validate_migration(self, group_handle, group_name, expected_state, expected_node):
    state_info = self._clusapi_utils.get_cluster_group_state(group_handle)
    owner_node = state_info['owner_node']
    group_state = state_info['state']
    if expected_state != group_state or expected_node.lower() != owner_node.lower():
        raise exceptions.ClusterGroupMigrationFailed(group_name=group_name, expected_state=expected_state, expected_node=expected_node, group_state=group_state, owner_node=owner_node)