import ctypes
import os
from unittest import mock
import uuid
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.virtdisk import vhdutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def _mock_open(self, read_data=None, curr_f_pos=0):
    mock_open = mock.mock_open()
    mock.patch.object(vhdutils, 'open', mock_open, create=True).start()
    f = mock_open.return_value
    f.read.side_effect = read_data
    f.tell.return_value = curr_f_pos
    return mock_open