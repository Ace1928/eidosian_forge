import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def _getoverviewfmt(self):
    """Internal: get the overview format. Queries the server if not
        already done, else returns the cached value."""
    try:
        return self._cachedoverviewfmt
    except AttributeError:
        pass
    try:
        resp, lines = self._longcmdstring('LIST OVERVIEW.FMT')
    except NNTPPermanentError:
        fmt = _DEFAULT_OVERVIEW_FMT[:]
    else:
        fmt = _parse_overview_fmt(lines)
    self._cachedoverviewfmt = fmt
    return fmt