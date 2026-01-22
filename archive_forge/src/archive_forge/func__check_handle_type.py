import contextlib
import ctypes
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def _check_handle_type(self, handle_type):
    if handle_type not in self._HANDLE_TYPES:
        err_msg = _('Invalid cluster handle type: %(handle_type)s. Allowed handle types: %(allowed_types)s.')
        raise exceptions.Invalid(err_msg % dict(handle_type=handle_type, allowed_types=self._HANDLE_TYPES))