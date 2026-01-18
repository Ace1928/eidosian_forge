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
def test_worker_finalization_via_atexit_handler_of_multiprocessing(self):
    cmd = 'if 1:\n            from multiprocess import Pool\n            problem = None\n            class A:\n                def __init__(self):\n                    self.pool = Pool(processes=1)\n            def test():\n                global problem\n                problem = A()\n                problem.pool.map(float, tuple(range(10)))\n            if __name__ == "__main__":\n                test()\n        '
    rc, out, err = test.support.script_helper.assert_python_ok('-c', cmd, **ENV)
    self.assertEqual(rc, 0)