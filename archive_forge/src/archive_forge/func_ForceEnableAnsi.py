from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
def ForceEnableAnsi():
    """Attempts to enable virtual terminal processing on Windows.

  Returns:
    bool: True if ANSI support is now active; False otherwise.
  """
    if platforms.OperatingSystem.Current() != platforms.OperatingSystem.WINDOWS:
        return False
    try:
        import ctypes
        enable_virtual_terminal_processing = 4
        h = ctypes.windll.kernel32.GetStdHandle(-11)
        old_mode = ctypes.wintypes.DWORD()
        if ctypes.windll.kernel32.GetConsoleMode(h, ctypes.byref(old_mode)):
            if ctypes.windll.kernel32.SetConsoleMode(h, old_mode.value | enable_virtual_terminal_processing):
                return True
    except (OSError, AttributeError):
        pass
    return False