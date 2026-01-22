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
def _kill_process(self, meth):
    if self.TYPE == 'threads':
        self.skipTest('test not appropriate for {}'.format(self.TYPE))
    p = self.Process(target=self._sleep_some)
    p.daemon = True
    p.start()
    self.assertEqual(p.is_alive(), True)
    self.assertIn(p, self.active_children())
    self.assertEqual(p.exitcode, None)
    join = TimingWrapper(p.join)
    self.assertEqual(join(0), None)
    self.assertTimingAlmostEqual(join.elapsed, 0.0)
    self.assertEqual(p.is_alive(), True)
    self.assertEqual(join(-1), None)
    self.assertTimingAlmostEqual(join.elapsed, 0.0)
    self.assertEqual(p.is_alive(), True)
    time.sleep(1)
    meth(p)
    if hasattr(signal, 'alarm'):

        def handler(*args):
            raise RuntimeError('join took too long: %s' % p)
        old_handler = signal.signal(signal.SIGALRM, handler)
        try:
            signal.alarm(10)
            self.assertEqual(join(), None)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        self.assertEqual(join(), None)
    self.assertTimingAlmostEqual(join.elapsed, 0.0)
    self.assertEqual(p.is_alive(), False)
    self.assertNotIn(p, self.active_children())
    p.join()
    return p.exitcode