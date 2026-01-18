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
def test_thread_safety(self):

    def cb():
        pass

    class Foo(object):

        def __init__(self):
            self.ref = self
            util.Finalize(self, cb, exitpriority=random.randint(1, 100))
    finish = False
    exc = None

    def run_finalizers():
        nonlocal exc
        while not finish:
            time.sleep(random.random() * 0.1)
            try:
                util._run_finalizers()
            except Exception as e:
                exc = e

    def make_finalizers():
        nonlocal exc
        d = {}
        while not finish:
            try:
                d[random.getrandbits(5)] = {Foo() for i in range(10)}
            except Exception as e:
                exc = e
                d.clear()
    old_interval = sys.getswitchinterval()
    old_threshold = gc.get_threshold()
    try:
        sys.setswitchinterval(1e-06)
        gc.set_threshold(5, 5, 5)
        threads = [threading.Thread(target=run_finalizers), threading.Thread(target=make_finalizers)]
        with threading_helper.start_threads(threads):
            time.sleep(4.0)
            finish = True
        if exc is not None:
            raise exc
    finally:
        sys.setswitchinterval(old_interval)
        gc.set_threshold(*old_threshold)
        gc.collect()