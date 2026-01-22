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
def _check_close(self):
    conn = self.headers.get('connection')
    if self.version == 11:
        if conn and 'close' in conn.lower():
            return True
        return False
    if self.headers.get('keep-alive'):
        return False
    if conn and 'keep-alive' in conn.lower():
        return False
    pconn = self.headers.get('proxy-connection')
    if pconn and 'keep-alive' in pconn.lower():
        return False
    return True