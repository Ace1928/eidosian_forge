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
@unittest.skipIf(os.name != 'posix', 'not feasible in non-posix platforms')
def test_shared_memory_SharedMemoryServer_ignores_sigint(self):
    smm = multiprocessing.managers.SharedMemoryManager()
    smm.start()
    sl = smm.ShareableList(range(10))
    os.kill(smm._process.pid, signal.SIGINT)
    sl2 = smm.ShareableList(range(10))
    with self.assertRaises(KeyboardInterrupt):
        os.kill(os.getpid(), signal.SIGINT)
    smm.shutdown()