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
def test_wait_integer(self):
    from multiprocess.connection import wait
    expected = 3
    sorted_ = lambda l: sorted(l, key=lambda x: id(x))
    sem = multiprocessing.Semaphore(0)
    a, b = multiprocessing.Pipe()
    p = multiprocessing.Process(target=self.signal_and_sleep, args=(sem, expected))
    p.start()
    self.assertIsInstance(p.sentinel, int)
    self.assertTrue(sem.acquire(timeout=20))
    start = time.monotonic()
    res = wait([a, p.sentinel, b], expected + 20)
    delta = time.monotonic() - start
    self.assertEqual(res, [p.sentinel])
    self.assertLess(delta, expected + 2)
    self.assertGreater(delta, expected - 2)
    a.send(None)
    start = time.monotonic()
    res = wait([a, p.sentinel, b], 20)
    delta = time.monotonic() - start
    self.assertEqual(sorted_(res), sorted_([p.sentinel, b]))
    self.assertLess(delta, 0.4)
    b.send(None)
    start = time.monotonic()
    res = wait([a, p.sentinel, b], 20)
    delta = time.monotonic() - start
    self.assertEqual(sorted_(res), sorted_([a, p.sentinel, b]))
    self.assertLess(delta, 0.4)
    p.terminate()
    p.join()