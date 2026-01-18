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
def test_queue_feeder_donot_stop_onexc(self):
    if self.TYPE != 'processes':
        self.skipTest('test not appropriate for {}'.format(self.TYPE))

    class NotSerializable(object):

        def __reduce__(self):
            raise AttributeError
    with test.support.captured_stderr():
        q = self.Queue()
        q.put(NotSerializable())
        q.put(True)
        self.assertTrue(q.get(timeout=support.SHORT_TIMEOUT))
        close_queue(q)
    with test.support.captured_stderr():
        q = self.Queue(maxsize=1)
        q.put(NotSerializable())
        q.put(True)
        try:
            self.assertEqual(q.qsize(), 1)
        except NotImplementedError:
            pass
        self.assertTrue(q.get(timeout=support.SHORT_TIMEOUT))
        self.assertTrue(q.empty())
        close_queue(q)