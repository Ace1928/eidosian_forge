import ctypes
import struct
from eventlet import patcher
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi import wintypes
@staticmethod
def write_buffer_data(buff, data):
    for i, c in enumerate(data):
        buff[i] = struct.unpack('B', six.b(c))[0]