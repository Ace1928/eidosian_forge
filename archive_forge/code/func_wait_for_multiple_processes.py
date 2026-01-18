import ctypes
from oslo_log import log as logging
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import kernel32 as kernel32_struct
def wait_for_multiple_processes(self, pids, wait_all=True, milliseconds=w_const.INFINITE):
    handles = []
    try:
        for pid in pids:
            handle = self.open_process(pid, desired_access=w_const.SYNCHRONIZE)
            handles.append(handle)
        return self._win32_utils.wait_for_multiple_objects(handles, wait_all, milliseconds)
    finally:
        for handle in handles:
            self._win32_utils.close_handle(handle)