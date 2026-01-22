from __future__ import (absolute_import, division, print_function)
import re
def _normalize_os(os_str):
    os_str = os_str.lower()
    if os_str == 'macos':
        os_str = 'darwin'
    return os_str