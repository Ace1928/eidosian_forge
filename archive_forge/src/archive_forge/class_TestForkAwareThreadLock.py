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
class TestForkAwareThreadLock(unittest.TestCase):

    @classmethod
    def child(cls, n, conn):
        if n > 1:
            p = multiprocessing.Process(target=cls.child, args=(n - 1, conn))
            p.start()
            conn.close()
            join_process(p)
        else:
            conn.send(len(util._afterfork_registry))
        conn.close()

    def test_lock(self):
        r, w = multiprocessing.Pipe(False)
        l = util.ForkAwareThreadLock()
        old_size = len(util._afterfork_registry)
        p = multiprocessing.Process(target=self.child, args=(5, w))
        p.start()
        w.close()
        new_size = r.recv()
        join_process(p)
        self.assertLessEqual(new_size, old_size)