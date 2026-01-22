import ctypes
import platform
import sys
from pprint import pformat as pformat_
from typing import Any
from packaging.version import Version as parse_version
def _tty_supports_color() -> bool:
    if sys.platform != 'win32':
        return True
    if parse_version(platform.version()) < parse_version('10.0.14393'):
        return True
    return _enable_windows_terminal_processing()