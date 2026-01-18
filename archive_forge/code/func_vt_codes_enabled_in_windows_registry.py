import functools
import os
import sys
from django.utils import termcolors
def vt_codes_enabled_in_windows_registry():
    """
        Check the Windows Registry to see if VT code handling has been enabled
        by default, see https://superuser.com/a/1300251/447564.
        """
    try:
        import winreg
    except ImportError:
        return False
    else:
        try:
            reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Console')
            reg_key_value, _ = winreg.QueryValueEx(reg_key, 'VirtualTerminalLevel')
        except FileNotFoundError:
            return False
        else:
            return reg_key_value == 1