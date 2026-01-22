from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
@property
def _hostinfo(self):
    netloc = self.netloc
    _, _, hostinfo = netloc.rpartition(b'@')
    _, have_open_br, bracketed = hostinfo.partition(b'[')
    if have_open_br:
        hostname, _, port = bracketed.partition(b']')
        _, _, port = port.partition(b':')
    else:
        hostname, _, port = hostinfo.partition(b':')
    if not port:
        port = None
    return (hostname, port)