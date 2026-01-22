import errno
import os
import re
import subprocess
import sys
import glob
def _RegistryGetValueUsingWinReg(key, value):
    """Use the _winreg module to obtain the value of a registry key.

  Args:
    key: The registry key.
    value: The particular registry value to read.
  Return:
    contents of the registry key's value, or None on failure.  Throws
    ImportError if winreg is unavailable.
  """
    from winreg import HKEY_LOCAL_MACHINE, OpenKey, QueryValueEx
    try:
        root, subkey = key.split('\\', 1)
        assert root == 'HKLM'
        with OpenKey(HKEY_LOCAL_MACHINE, subkey) as hkey:
            return QueryValueEx(hkey, value)[0]
    except OSError:
        return None