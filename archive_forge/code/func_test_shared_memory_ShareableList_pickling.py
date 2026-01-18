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
def test_shared_memory_ShareableList_pickling(self):
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        with self.subTest(proto=proto):
            sl = shared_memory.ShareableList(range(10))
            self.addCleanup(sl.shm.unlink)
            serialized_sl = pickle.dumps(sl, protocol=proto)
            deserialized_sl = pickle.loads(serialized_sl)
            self.assertIsInstance(deserialized_sl, shared_memory.ShareableList)
            self.assertEqual(deserialized_sl[-1], 9)
            self.assertIsNot(sl, deserialized_sl)
            deserialized_sl[4] = 'changed'
            self.assertEqual(sl[4], 'changed')
            sl[3] = 'newvalue'
            self.assertEqual(deserialized_sl[3], 'newvalue')
            larger_sl = shared_memory.ShareableList(range(400))
            self.addCleanup(larger_sl.shm.unlink)
            serialized_larger_sl = pickle.dumps(larger_sl, protocol=proto)
            self.assertEqual(len(serialized_sl), len(serialized_larger_sl))
            larger_sl.shm.close()
            deserialized_sl.shm.close()
            sl.shm.close()