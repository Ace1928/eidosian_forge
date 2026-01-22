import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def _get_numa_cpu_info(self, numa_node_assoc, processors):
    cpu_info = []
    paths = [x.path_().upper() for x in numa_node_assoc]
    for proc in processors:
        if proc.path_().upper() in paths:
            cpu_info.append(proc)
    return cpu_info