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
def test_preload_resources(self):
    if multiprocessing.get_start_method() != 'forkserver':
        self.skipTest("test only relevant for 'forkserver' method")
    name = os.path.join(os.path.dirname(__file__), 'mp_preload.py')
    rc, out, err = test.support.script_helper.assert_python_ok(name, **ENV)
    out = out.decode()
    err = err.decode()
    if out.rstrip() != 'ok' or err != '':
        print(out)
        print(err)
        self.fail('failed spawning forkserver or grandchild')