import email.parser
import email.message
import errno
import http
import io
import re
import socket
import sys
import collections.abc
from urllib.parse import urlsplit
def _read_status(self):
    line = str(self.fp.readline(_MAXLINE + 1), 'iso-8859-1')
    if len(line) > _MAXLINE:
        raise LineTooLong('status line')
    if self.debuglevel > 0:
        print('reply:', repr(line))
    if not line:
        raise RemoteDisconnected('Remote end closed connection without response')
    try:
        version, status, reason = line.split(None, 2)
    except ValueError:
        try:
            version, status = line.split(None, 1)
            reason = ''
        except ValueError:
            version = ''
    if not version.startswith('HTTP/'):
        self._close_conn()
        raise BadStatusLine(line)
    try:
        status = int(status)
        if status < 100 or status > 999:
            raise BadStatusLine(line)
    except ValueError:
        raise BadStatusLine(line)
    return (version, status, reason)