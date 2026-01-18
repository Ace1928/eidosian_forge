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
def test_shared_memory_pickle_unpickle(self):
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        with self.subTest(proto=proto):
            sms = shared_memory.SharedMemory(create=True, size=512)
            self.addCleanup(sms.unlink)
            sms.buf[0:6] = b'pickle'
            pickled_sms = pickle.dumps(sms, protocol=proto)
            sms2 = pickle.loads(pickled_sms)
            self.assertIsInstance(sms2, shared_memory.SharedMemory)
            self.assertEqual(sms.name, sms2.name)
            self.assertEqual(bytes(sms.buf[0:6]), b'pickle')
            self.assertEqual(bytes(sms2.buf[0:6]), b'pickle')
            sms.buf[0:6] = b'newval'
            self.assertEqual(bytes(sms.buf[0:6]), b'newval')
            self.assertEqual(bytes(sms2.buf[0:6]), b'newval')
            sms2.buf[0:6] = b'oldval'
            self.assertEqual(bytes(sms.buf[0:6]), b'oldval')
            self.assertEqual(bytes(sms2.buf[0:6]), b'oldval')