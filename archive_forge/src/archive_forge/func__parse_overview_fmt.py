import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def _parse_overview_fmt(lines):
    """Parse a list of string representing the response to LIST OVERVIEW.FMT
    and return a list of header/metadata names.
    Raises NNTPDataError if the response is not compliant
    (cf. RFC 3977, section 8.4)."""
    fmt = []
    for line in lines:
        if line[0] == ':':
            name, _, suffix = line[1:].partition(':')
            name = ':' + name
        else:
            name, _, suffix = line.partition(':')
        name = name.lower()
        name = _OVERVIEW_FMT_ALTERNATIVES.get(name, name)
        fmt.append(name)
    defaults = _DEFAULT_OVERVIEW_FMT
    if len(fmt) < len(defaults):
        raise NNTPDataError('LIST OVERVIEW.FMT response too short')
    if fmt[:len(defaults)] != defaults:
        raise NNTPDataError('LIST OVERVIEW.FMT redefines default fields')
    return fmt