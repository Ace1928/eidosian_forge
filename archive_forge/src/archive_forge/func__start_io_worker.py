import errno
import os
from eventlet import patcher
from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
def _start_io_worker(self, func, buff, overlapped_structure, completion_routine, buff_update_func=None):
    try:
        while not self._stopped.isSet():
            if buff_update_func:
                num_bytes = buff_update_func()
                if not num_bytes:
                    continue
            else:
                num_bytes = len(buff)
            func(self._pipe_handle, buff, num_bytes, overlapped_structure, completion_routine)
    except Exception:
        self._stopped.set()
    finally:
        with self._lock:
            self._cleanup_handles()