import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
def _get_match_result(address, formats):
    for regexp in formats:
        match = regexp.findall(address)
        if match:
            return match[0]