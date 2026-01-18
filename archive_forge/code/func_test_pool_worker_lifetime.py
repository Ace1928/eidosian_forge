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
def test_pool_worker_lifetime(self):
    p = multiprocessing.Pool(3, maxtasksperchild=10)
    self.assertEqual(3, len(p._pool))
    origworkerpids = [w.pid for w in p._pool]
    results = []
    for i in range(100):
        results.append(p.apply_async(sqr, (i,)))
    for j, res in enumerate(results):
        self.assertEqual(res.get(), sqr(j))
    p._repopulate_pool()
    countdown = 50
    while countdown and (not all((w.is_alive() for w in p._pool))):
        countdown -= 1
        time.sleep(DELTA)
    finalworkerpids = [w.pid for w in p._pool]
    self.assertNotIn(None, origworkerpids)
    self.assertNotIn(None, finalworkerpids)
    self.assertNotEqual(sorted(origworkerpids), sorted(finalworkerpids))
    p.close()
    p.join()