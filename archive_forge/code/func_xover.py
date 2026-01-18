import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def xover(self, start, end, *, file=None):
    """Process an XOVER command (optional server extension) Arguments:
        - start: start of range
        - end: end of range
        - file: Filename string or file object to store the result in
        Returns:
        - resp: server response if successful
        - list: list of dicts containing the response fields
        """
    resp, lines = self._longcmdstring('XOVER {0}-{1}'.format(start, end), file)
    fmt = self._getoverviewfmt()
    return (resp, _parse_overview(lines, fmt))