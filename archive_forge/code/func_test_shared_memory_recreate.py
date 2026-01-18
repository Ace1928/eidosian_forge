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
@unittest.skipIf(True, 'fails with dill >= 0.3.5')
def test_shared_memory_recreate(self):
    with unittest.mock.patch('multiprocess.shared_memory._make_filename') as mock_make_filename:
        NAME_PREFIX = shared_memory._SHM_NAME_PREFIX
        names = [self._new_shm_name('test03_fn'), self._new_shm_name('test04_fn')]
        names = [NAME_PREFIX + name for name in names]
        mock_make_filename.side_effect = names
        shm1 = shared_memory.SharedMemory(create=True, size=1)
        self.addCleanup(shm1.unlink)
        self.assertEqual(shm1._name, names[0])
        mock_make_filename.side_effect = names
        shm2 = shared_memory.SharedMemory(create=True, size=1)
        self.addCleanup(shm2.unlink)
        self.assertEqual(shm2._name, names[1])