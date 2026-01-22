import contextlib
import ctypes
import os
import shutil
import tempfile
from oslo_log import log as logging
from oslo_utils import fileutils
from os_win._i18n import _
from os_win import _utils
import os_win.conf
from os_win import exceptions
from os_win.utils import _acl_utils
from os_win.utils.io import ioutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import advapi32 as advapi32_def
from os_win.utils.winapi.libs import kernel32 as kernel32_def
from os_win.utils.winapi import wintypes
def add_acl_rule(self, path, trustee_name, access_rights, access_mode, inheritance_flags=0):
    """Adds the requested access rule to a file or object.

        Can be used for granting/revoking access.
        """
    p_to_free = []
    try:
        sec_info = self._acl_utils.get_named_security_info(obj_name=path, obj_type=w_const.SE_FILE_OBJECT, security_info_flags=w_const.DACL_SECURITY_INFORMATION)
        p_to_free.append(sec_info['pp_sec_desc'].contents)
        access = advapi32_def.EXPLICIT_ACCESS()
        access.grfAccessPermissions = access_rights
        access.grfAccessMode = access_mode
        access.grfInheritance = inheritance_flags
        access.Trustee.TrusteeForm = w_const.TRUSTEE_IS_NAME
        access.Trustee.pstrName = ctypes.c_wchar_p(trustee_name)
        pp_new_dacl = self._acl_utils.set_entries_in_acl(entry_count=1, p_explicit_entry_list=ctypes.pointer(access), p_old_acl=sec_info['pp_dacl'].contents)
        p_to_free.append(pp_new_dacl.contents)
        self._acl_utils.set_named_security_info(obj_name=path, obj_type=w_const.SE_FILE_OBJECT, security_info_flags=w_const.DACL_SECURITY_INFORMATION, p_dacl=pp_new_dacl.contents)
    finally:
        for p in p_to_free:
            self._win32_utils.local_free(p)