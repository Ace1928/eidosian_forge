import errno
import os
from eventlet import patcher
from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
def _setup_io_structures(self):
    self._r_buffer = self._ioutils.get_buffer(constants.SERIAL_CONSOLE_BUFFER_SIZE)
    self._w_buffer = self._ioutils.get_buffer(constants.SERIAL_CONSOLE_BUFFER_SIZE)
    self._r_overlapped = self._ioutils.get_new_overlapped_structure()
    self._w_overlapped = self._ioutils.get_new_overlapped_structure()
    self._r_completion_routine = self._ioutils.get_completion_routine(self._read_callback)
    self._w_completion_routine = self._ioutils.get_completion_routine()
    self._log_file_handle = None