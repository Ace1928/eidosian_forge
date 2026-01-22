from __future__ import division
from __future__ import print_function
import collections
import contextlib
import errno
import functools
import os
import socket
import stat
import sys
import threading
import warnings
from collections import namedtuple
from socket import AF_INET
from socket import SOCK_DGRAM
from socket import SOCK_STREAM
def _infodict(self, attrs):
    info = collections.OrderedDict()
    for name in attrs:
        value = getattr(self, name, None)
        if value:
            info[name] = value
        elif name == 'pid' and value == 0:
            info[name] = value
    return info