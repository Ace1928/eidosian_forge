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
def test_put(self):
    MAXSIZE = 6
    queue = self.Queue(maxsize=MAXSIZE)
    child_can_start = self.Event()
    parent_can_continue = self.Event()
    proc = self.Process(target=self._test_put, args=(queue, child_can_start, parent_can_continue))
    proc.daemon = True
    proc.start()
    self.assertEqual(queue_empty(queue), True)
    self.assertEqual(queue_full(queue, MAXSIZE), False)
    queue.put(1)
    queue.put(2, True)
    queue.put(3, True, None)
    queue.put(4, False)
    queue.put(5, False, None)
    queue.put_nowait(6)
    time.sleep(DELTA)
    self.assertEqual(queue_empty(queue), False)
    self.assertEqual(queue_full(queue, MAXSIZE), True)
    put = TimingWrapper(queue.put)
    put_nowait = TimingWrapper(queue.put_nowait)
    self.assertRaises(pyqueue.Full, put, 7, False)
    self.assertTimingAlmostEqual(put.elapsed, 0)
    self.assertRaises(pyqueue.Full, put, 7, False, None)
    self.assertTimingAlmostEqual(put.elapsed, 0)
    self.assertRaises(pyqueue.Full, put_nowait, 7)
    self.assertTimingAlmostEqual(put_nowait.elapsed, 0)
    self.assertRaises(pyqueue.Full, put, 7, True, TIMEOUT1)
    self.assertTimingAlmostEqual(put.elapsed, TIMEOUT1)
    self.assertRaises(pyqueue.Full, put, 7, False, TIMEOUT2)
    self.assertTimingAlmostEqual(put.elapsed, 0)
    self.assertRaises(pyqueue.Full, put, 7, True, timeout=TIMEOUT3)
    self.assertTimingAlmostEqual(put.elapsed, TIMEOUT3)
    child_can_start.set()
    parent_can_continue.wait()
    self.assertEqual(queue_empty(queue), True)
    self.assertEqual(queue_full(queue, MAXSIZE), False)
    proc.join()
    close_queue(queue)