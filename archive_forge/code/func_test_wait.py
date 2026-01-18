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
def test_wait(self, slow=False):
    from multiprocess.connection import wait
    readers = []
    procs = []
    messages = []
    for i in range(4):
        r, w = multiprocessing.Pipe(duplex=False)
        p = multiprocessing.Process(target=self._child_test_wait, args=(w, slow))
        p.daemon = True
        p.start()
        w.close()
        readers.append(r)
        procs.append(p)
        self.addCleanup(p.join)
    while readers:
        for r in wait(readers):
            try:
                msg = r.recv()
            except EOFError:
                readers.remove(r)
                r.close()
            else:
                messages.append(msg)
    messages.sort()
    expected = sorted(((i, p.pid) for i in range(10) for p in procs))
    self.assertEqual(messages, expected)