import collections
import collections.abc
import concurrent.futures
import errno
import functools
import heapq
import itertools
import os
import socket
import stat
import subprocess
import threading
import time
import traceback
import sys
import warnings
import weakref
from . import constants
from . import coroutines
from . import events
from . import exceptions
from . import futures
from . import protocols
from . import sslproto
from . import staggered
from . import tasks
from . import transports
from . import trsock
from .log import logger
def _ipaddr_info(host, port, family, type, proto, flowinfo=0, scopeid=0):
    if not hasattr(socket, 'inet_pton'):
        return
    if proto not in {0, socket.IPPROTO_TCP, socket.IPPROTO_UDP} or host is None:
        return None
    if type == socket.SOCK_STREAM:
        proto = socket.IPPROTO_TCP
    elif type == socket.SOCK_DGRAM:
        proto = socket.IPPROTO_UDP
    else:
        return None
    if port is None:
        port = 0
    elif isinstance(port, bytes) and port == b'':
        port = 0
    elif isinstance(port, str) and port == '':
        port = 0
    else:
        try:
            port = int(port)
        except (TypeError, ValueError):
            return None
    if family == socket.AF_UNSPEC:
        afs = [socket.AF_INET]
        if _HAS_IPv6:
            afs.append(socket.AF_INET6)
    else:
        afs = [family]
    if isinstance(host, bytes):
        host = host.decode('idna')
    if '%' in host:
        return None
    for af in afs:
        try:
            socket.inet_pton(af, host)
            if _HAS_IPv6 and af == socket.AF_INET6:
                return (af, type, proto, '', (host, port, flowinfo, scopeid))
            else:
                return (af, type, proto, '', (host, port))
        except OSError:
            pass
    return None