import unittest
import unittest.mock
import queue as pyqueue
import textwrap
import time
import io
import itertools
import sys
import os
import gc
import errno
import signal
import array
import socket
import random
import logging
import subprocess
import struct
import operator
import pickle #XXX: use dill?
import weakref
import warnings
import test.support
import test.support.script_helper
from test import support
from test.support import hashlib_helper
from test.support import import_helper
from test.support import os_helper
from test.support import socket_helper
from test.support import threading_helper
from test.support import warnings_helper
import_helper.import_module('multiprocess.synchronize')
import threading
import multiprocess as multiprocessing
import multiprocess.connection
import multiprocess.dummy
import multiprocess.heap
import multiprocess.managers
import multiprocess.pool
import multiprocess.queues
from multiprocess import util
from multiprocess.connection import wait
from multiprocess.managers import BaseManager, BaseProxy, RemoteError
def test_wait_socket(self, slow=False):
    from multiprocess.connection import wait
    l = socket.create_server((socket_helper.HOST, 0))
    addr = l.getsockname()
    readers = []
    procs = []
    dic = {}
    for i in range(4):
        p = multiprocessing.Process(target=self._child_test_wait_socket, args=(addr, slow))
        p.daemon = True
        p.start()
        procs.append(p)
        self.addCleanup(p.join)
    for i in range(4):
        r, _ = l.accept()
        readers.append(r)
        dic[r] = []
    l.close()
    while readers:
        for r in wait(readers):
            msg = r.recv(32)
            if not msg:
                readers.remove(r)
                r.close()
            else:
                dic[r].append(msg)
    expected = ''.join(('%s\n' % i for i in range(10))).encode('ascii')
    for v in dic.values():
        self.assertEqual(b''.join(v), expected)