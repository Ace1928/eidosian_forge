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
class _TestArray(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    @classmethod
    def f(cls, seq):
        for i in range(1, len(seq)):
            seq[i] += seq[i - 1]

    @unittest.skipIf(c_int is None, 'requires _ctypes')
    def test_array(self, raw=False):
        seq = [680, 626, 934, 821, 150, 233, 548, 982, 714, 831]
        if raw:
            arr = self.RawArray('i', seq)
        else:
            arr = self.Array('i', seq)
        self.assertEqual(len(arr), len(seq))
        self.assertEqual(arr[3], seq[3])
        self.assertEqual(list(arr[2:7]), list(seq[2:7]))
        arr[4:8] = seq[4:8] = array.array('i', [1, 2, 3, 4])
        self.assertEqual(list(arr[:]), seq)
        self.f(seq)
        p = self.Process(target=self.f, args=(arr,))
        p.daemon = True
        p.start()
        p.join()
        self.assertEqual(list(arr[:]), seq)

    @unittest.skipIf(c_int is None, 'requires _ctypes')
    def test_array_from_size(self):
        size = 10
        for _ in range(3):
            arr = self.Array('i', size)
            self.assertEqual(len(arr), size)
            self.assertEqual(list(arr), [0] * size)
            arr[:] = range(10)
            self.assertEqual(list(arr), list(range(10)))
            del arr

    @unittest.skipIf(c_int is None, 'requires _ctypes')
    def test_rawarray(self):
        self.test_array(raw=True)

    @unittest.skipIf(c_int is None, 'requires _ctypes')
    def test_getobj_getlock_obj(self):
        arr1 = self.Array('i', list(range(10)))
        lock1 = arr1.get_lock()
        obj1 = arr1.get_obj()
        arr2 = self.Array('i', list(range(10)), lock=None)
        lock2 = arr2.get_lock()
        obj2 = arr2.get_obj()
        lock = self.Lock()
        arr3 = self.Array('i', list(range(10)), lock=lock)
        lock3 = arr3.get_lock()
        obj3 = arr3.get_obj()
        self.assertEqual(lock, lock3)
        arr4 = self.Array('i', range(10), lock=False)
        self.assertFalse(hasattr(arr4, 'get_lock'))
        self.assertFalse(hasattr(arr4, 'get_obj'))
        self.assertRaises(AttributeError, self.Array, 'i', range(10), lock='notalock')
        arr5 = self.RawArray('i', range(10))
        self.assertFalse(hasattr(arr5, 'get_lock'))
        self.assertFalse(hasattr(arr5, 'get_obj'))