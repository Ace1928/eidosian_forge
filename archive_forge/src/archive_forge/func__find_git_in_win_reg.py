import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def _find_git_in_win_reg():
    import platform
    import winreg
    if platform.machine() == 'AMD64':
        subkey = 'SOFTWARE\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Git_is1'
    else:
        subkey = 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Git_is1'
    for key in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        with suppress(OSError):
            with winreg.OpenKey(key, subkey) as k:
                val, typ = winreg.QueryValueEx(k, 'InstallLocation')
                if typ == winreg.REG_SZ:
                    yield val