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
class _TestPoll(BaseTestCase):
    ALLOWED_TYPES = ('processes', 'threads')

    def test_empty_string(self):
        a, b = self.Pipe()
        self.assertEqual(a.poll(), False)
        b.send_bytes(b'')
        self.assertEqual(a.poll(), True)
        self.assertEqual(a.poll(), True)

    @classmethod
    def _child_strings(cls, conn, strings):
        for s in strings:
            time.sleep(0.1)
            conn.send_bytes(s)
        conn.close()

    def test_strings(self):
        strings = (b'hello', b'', b'a', b'b', b'', b'bye', b'', b'lop')
        a, b = self.Pipe()
        p = self.Process(target=self._child_strings, args=(b, strings))
        p.start()
        for s in strings:
            for i in range(200):
                if a.poll(0.01):
                    break
            x = a.recv_bytes()
            self.assertEqual(s, x)
        p.join()

    @classmethod
    def _child_boundaries(cls, r):
        r.poll(5)

    def test_boundaries(self):
        r, w = self.Pipe(False)
        p = self.Process(target=self._child_boundaries, args=(r,))
        p.start()
        time.sleep(2)
        L = [b'first', b'second']
        for obj in L:
            w.send_bytes(obj)
        w.close()
        p.join()
        self.assertIn(r.recv_bytes(), L)

    @classmethod
    def _child_dont_merge(cls, b):
        b.send_bytes(b'a')
        b.send_bytes(b'b')
        b.send_bytes(b'cd')

    def test_dont_merge(self):
        a, b = self.Pipe()
        self.assertEqual(a.poll(0.0), False)
        self.assertEqual(a.poll(0.1), False)
        p = self.Process(target=self._child_dont_merge, args=(b,))
        p.start()
        self.assertEqual(a.recv_bytes(), b'a')
        self.assertEqual(a.poll(1.0), True)
        self.assertEqual(a.poll(1.0), True)
        self.assertEqual(a.recv_bytes(), b'b')
        self.assertEqual(a.poll(1.0), True)
        self.assertEqual(a.poll(1.0), True)
        self.assertEqual(a.poll(0.0), True)
        self.assertEqual(a.recv_bytes(), b'cd')
        p.join()