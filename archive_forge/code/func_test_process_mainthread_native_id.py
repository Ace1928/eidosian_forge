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
@unittest.skipUnless(threading._HAVE_THREAD_NATIVE_ID, 'needs native_id')
def test_process_mainthread_native_id(self):
    if self.TYPE == 'threads':
        self.skipTest('test not appropriate for {}'.format(self.TYPE))
    current_mainthread_native_id = threading.main_thread().native_id
    q = self.Queue(1)
    p = self.Process(target=self._test_process_mainthread_native_id, args=(q,))
    p.start()
    child_mainthread_native_id = q.get()
    p.join()
    close_queue(q)
    self.assertNotEqual(current_mainthread_native_id, child_mainthread_native_id)