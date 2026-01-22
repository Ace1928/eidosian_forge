import errno
import inspect
import json
import logging
import os
import re
import select
import socket
import sys
import time
import traceback
import uuid
from pdb import Pdb
from typing import Callable
import setproctitle
import ray
from ray._private import ray_constants
from ray.experimental.internal_kv import _internal_kv_del, _internal_kv_put
from ray.util.annotations import DeveloperAPI
class _LF2CRLF_FileWrapper(object):

    def __init__(self, connection):
        self.connection = connection
        self.stream = fh = connection.makefile('rw')
        self.read = fh.read
        self.readline = fh.readline
        self.readlines = fh.readlines
        self.close = fh.close
        self.flush = fh.flush
        self.fileno = fh.fileno
        if hasattr(fh, 'encoding'):
            self._send = lambda data: connection.sendall(data.encode(fh.encoding))
        else:
            self._send = connection.sendall

    @property
    def encoding(self):
        return self.stream.encoding

    def __iter__(self):
        return self.stream.__iter__()

    def write(self, data, nl_rex=re.compile('\r?\n')):
        data = nl_rex.sub('\r\n', data)
        self._send(data)

    def writelines(self, lines, nl_rex=re.compile('\r?\n')):
        for line in lines:
            self.write(line, nl_rex)