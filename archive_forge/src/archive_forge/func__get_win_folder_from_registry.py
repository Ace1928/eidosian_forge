import sys
import os
def _get_win_folder_from_registry(csidl_name):
    """This is a fallback technique at best. I'm not sure if using the
    registry for this guarantees us the correct answer for all CSIDL_*
    names.
    """
    import winreg as _winreg
    shell_folder_name = {'CSIDL_APPDATA': 'AppData', 'CSIDL_COMMON_APPDATA': 'Common AppData', 'CSIDL_LOCAL_APPDATA': 'Local AppData'}[csidl_name]
    key = _winreg.OpenKey(_winreg.HKEY_CURRENT_USER, 'Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders')
    dir, type = _winreg.QueryValueEx(key, shell_folder_name)
    return dir