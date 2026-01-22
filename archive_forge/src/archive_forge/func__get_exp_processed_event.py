import ctypes
from unittest import mock
import ddt
from six.moves import queue
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import clusterutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def _get_exp_processed_event(self, event, **kwargs):
    preserved_keys = ['cluster_object_name', 'object_type', 'filter_flags', 'notif_key']
    exp_proc_evt = {key: event[key] for key in preserved_keys}
    exp_proc_evt.update(**kwargs)
    return exp_proc_evt