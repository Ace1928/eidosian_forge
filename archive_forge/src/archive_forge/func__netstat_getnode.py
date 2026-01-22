import os
import sys
from enum import Enum, _simple_enum
def _netstat_getnode():
    """Get the hardware address on Unix by running netstat."""
    return _find_mac_under_heading('netstat', '-ian', b'Address')