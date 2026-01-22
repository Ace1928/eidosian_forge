import ctypes
from oslo_log import log as logging
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import kernel32 as kernel32_struct
class Mutex(object):

    def __init__(self, name=None):
        self.name = name
        self._processutils = ProcessUtils()
        self._win32_utils = win32utils.Win32Utils()
        self._handle = self._processutils.create_mutex(self.name)

    def acquire(self, timeout_ms=w_const.INFINITE):
        try:
            self._win32_utils.wait_for_single_object(self._handle, timeout_ms)
            return True
        except exceptions.Timeout:
            return False

    def release(self):
        self._processutils.release_mutex(self._handle)

    def close(self):
        if self._handle:
            self._win32_utils.close_handle(self._handle)
        self._handle = None
    __del__ = close

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()