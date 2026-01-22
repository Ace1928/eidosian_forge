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
class _TestSharedCTypes(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    def setUp(self):
        if not HAS_SHAREDCTYPES:
            self.skipTest('requires multiprocess.sharedctypes')

    @classmethod
    def _double(cls, x, y, z, foo, arr, string):
        x.value *= 2
        y.value *= 2
        z.value *= 2
        foo.x *= 2
        foo.y *= 2
        string.value *= 2
        for i in range(len(arr)):
            arr[i] *= 2

    def test_sharedctypes(self, lock=False):
        x = Value('i', 7, lock=lock)
        y = Value(c_double, 1.0 / 3.0, lock=lock)
        z = Value(c_longlong, 2 ** 33, lock=lock)
        foo = Value(_Foo, 3, 2, lock=lock)
        arr = self.Array('d', list(range(10)), lock=lock)
        string = self.Array('c', 20, lock=lock)
        string.value = latin('hello')
        p = self.Process(target=self._double, args=(x, y, z, foo, arr, string))
        p.daemon = True
        p.start()
        p.join()
        self.assertEqual(x.value, 14)
        self.assertAlmostEqual(y.value, 2.0 / 3.0)
        self.assertEqual(z.value, 2 ** 34)
        self.assertEqual(foo.x, 6)
        self.assertAlmostEqual(foo.y, 4.0)
        for i in range(10):
            self.assertAlmostEqual(arr[i], i * 2)
        self.assertEqual(string.value, latin('hellohello'))

    def test_synchronize(self):
        self.test_sharedctypes(lock=True)

    def test_copy(self):
        foo = _Foo(2, 5.0, 2 ** 33)
        bar = copy(foo)
        foo.x = 0
        foo.y = 0
        foo.z = 0
        self.assertEqual(bar.x, 2)
        self.assertAlmostEqual(bar.y, 5.0)
        self.assertEqual(bar.z, 2 ** 33)