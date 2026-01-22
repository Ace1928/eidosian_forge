import errno
import os
from eventlet import patcher
from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
def _retry_if_file_in_use(self, f, *args, **kwargs):
    retry_count = 0
    while True:
        try:
            return f(*args, **kwargs)
        except WindowsError as err:
            if err.errno == errno.EACCES and retry_count < self._MAX_LOG_ROTATE_RETRIES:
                retry_count += 1
                time.sleep(1)
            else:
                raise