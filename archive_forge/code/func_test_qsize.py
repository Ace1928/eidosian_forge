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
def test_qsize(self):
    q = self.Queue()
    try:
        self.assertEqual(q.qsize(), 0)
    except NotImplementedError:
        self.skipTest('qsize method not implemented')
    q.put(1)
    self.assertEqual(q.qsize(), 1)
    q.put(5)
    self.assertEqual(q.qsize(), 2)
    q.get()
    self.assertEqual(q.qsize(), 1)
    q.get()
    self.assertEqual(q.qsize(), 0)
    close_queue(q)