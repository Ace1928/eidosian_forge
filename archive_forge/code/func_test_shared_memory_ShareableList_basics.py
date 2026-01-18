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
def test_shared_memory_ShareableList_basics(self):
    sl = shared_memory.ShareableList(['howdy', b'HoWdY', -273.154, 100, None, True, 42])
    self.addCleanup(sl.shm.unlink)
    self.assertIn(sl.shm.name, str(sl))
    self.assertIn(str(list(sl)), str(sl))
    with self.assertRaises(IndexError):
        sl[7]
    with self.assertRaises(IndexError):
        sl[7] = 2
    current_format = sl._get_packing_format(0)
    sl[0] = 'howdy'
    self.assertEqual(current_format, sl._get_packing_format(0))
    self.assertEqual(sl.format, '8s8sdqxxxxxx?xxxxxxxx?q')
    self.assertEqual(len(sl), 7)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with self.assertRaises(ValueError):
            sl.index('100')
        self.assertEqual(sl.index(100), 3)
    self.assertEqual(sl[0], 'howdy')
    self.assertEqual(sl[-2], True)
    self.assertEqual(tuple(sl), ('howdy', b'HoWdY', -273.154, 100, None, True, 42))
    sl[3] = 42
    self.assertEqual(sl[3], 42)
    sl[4] = 'some'
    self.assertEqual(sl[4], 'some')
    self.assertEqual(sl.format, '8s8sdq8sxxxxxxx?q')
    with self.assertRaisesRegex(ValueError, 'exceeds available storage'):
        sl[4] = 'far too many'
    self.assertEqual(sl[4], 'some')
    sl[0] = 'encodés'
    self.assertEqual(sl[0], 'encodés')
    self.assertEqual(sl[1], b'HoWdY')
    with self.assertRaisesRegex(ValueError, 'exceeds available storage'):
        sl[0] = 'encodées'
    self.assertEqual(sl[1], b'HoWdY')
    with self.assertRaisesRegex(ValueError, 'exceeds available storage'):
        sl[1] = b'123456789'
    self.assertEqual(sl[1], b'HoWdY')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        self.assertEqual(sl.count(42), 2)
        self.assertEqual(sl.count(b'HoWdY'), 1)
        self.assertEqual(sl.count(b'adios'), 0)
    name_duplicate = self._new_shm_name('test03_duplicate')
    sl_copy = shared_memory.ShareableList(sl, name=name_duplicate)
    try:
        self.assertNotEqual(sl.shm.name, sl_copy.shm.name)
        self.assertEqual(name_duplicate, sl_copy.shm.name)
        self.assertEqual(list(sl), list(sl_copy))
        self.assertEqual(sl.format, sl_copy.format)
        sl_copy[-1] = 77
        self.assertEqual(sl_copy[-1], 77)
        self.assertNotEqual(sl[-1], 77)
        sl_copy.shm.close()
    finally:
        sl_copy.shm.unlink()
    sl_tethered = shared_memory.ShareableList(name=sl.shm.name)
    self.assertEqual(sl.shm.name, sl_tethered.shm.name)
    sl_tethered[-1] = 880
    self.assertEqual(sl[-1], 880)
    sl_tethered.shm.close()
    sl.shm.close()
    empty_sl = shared_memory.ShareableList()
    try:
        self.assertEqual(len(empty_sl), 0)
        self.assertEqual(empty_sl.format, '')
        self.assertEqual(empty_sl.count('any'), 0)
        with self.assertRaises(ValueError):
            empty_sl.index(None)
        empty_sl.shm.close()
    finally:
        empty_sl.shm.unlink()