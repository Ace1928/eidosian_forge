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
def test_notify_n(self):
    cond = self.Condition()
    sleeping = self.Semaphore(0)
    woken = self.Semaphore(0)
    for i in range(3):
        p = self.Process(target=self.f, args=(cond, sleeping, woken))
        p.daemon = True
        p.start()
        self.addCleanup(p.join)
        t = threading.Thread(target=self.f, args=(cond, sleeping, woken))
        t.daemon = True
        t.start()
        self.addCleanup(t.join)
    for i in range(6):
        sleeping.acquire()
    time.sleep(DELTA)
    self.assertReturnsIfImplemented(0, get_value, woken)
    cond.acquire()
    cond.notify(n=2)
    cond.release()
    self.assertReachesEventually(lambda: get_value(woken), 2)
    cond.acquire()
    cond.notify(n=4)
    cond.release()
    self.assertReachesEventually(lambda: get_value(woken), 6)
    cond.acquire()
    cond.notify(n=3)
    cond.release()
    self.assertReturnsIfImplemented(6, get_value, woken)
    self.check_invariant(cond)