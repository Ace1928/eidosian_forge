import errno
import os
import re
import subprocess
import sys
import glob
def _RegistryQuery(key, value=None):
    """Use reg.exe to read a particular key through _RegistryQueryBase.

  First tries to launch from %WinDir%\\Sysnative to avoid WoW64 redirection. If
  that fails, it falls back to System32.  Sysnative is available on Vista and
  up and available on Windows Server 2003 and XP through KB patch 942589. Note
  that Sysnative will always fail if using 64-bit python due to it being a
  virtual directory and System32 will work correctly in the first place.

  KB 942589 - http://support.microsoft.com/kb/942589/en-us.

  Arguments:
    key: The registry key.
    value: The particular registry value to read (optional).
  Return:
    stdout from reg.exe, or None for failure.
  """
    text = None
    try:
        text = _RegistryQueryBase('Sysnative', key, value)
    except OSError as e:
        if e.errno == errno.ENOENT:
            text = _RegistryQueryBase('System32', key, value)
        else:
            raise
    return text