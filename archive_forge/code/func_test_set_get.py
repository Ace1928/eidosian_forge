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
def test_set_get(self):
    multiprocessing.set_forkserver_preload(PRELOAD)
    count = 0
    old_method = multiprocessing.get_start_method()
    try:
        for method in ('fork', 'spawn', 'forkserver'):
            try:
                multiprocessing.set_start_method(method, force=True)
            except ValueError:
                continue
            self.assertEqual(multiprocessing.get_start_method(), method)
            ctx = multiprocessing.get_context()
            self.assertEqual(ctx.get_start_method(), method)
            self.assertTrue(type(ctx).__name__.lower().startswith(method))
            self.assertTrue(ctx.Process.__name__.lower().startswith(method))
            self.check_context(multiprocessing)
            count += 1
    finally:
        multiprocessing.set_start_method(old_method, force=True)
    self.assertGreaterEqual(count, 1)