import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
def update_buff(func, adapter_index, buff):
    buff.value = six.b(self._FAKE_ADAPTER_NAME)