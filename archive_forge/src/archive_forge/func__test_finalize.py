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
@classmethod
def _test_finalize(cls, conn):

    class Foo(object):
        pass
    a = Foo()
    util.Finalize(a, conn.send, args=('a',))
    del a
    gc.collect()
    b = Foo()
    close_b = util.Finalize(b, conn.send, args=('b',))
    close_b()
    close_b()
    del b
    gc.collect()
    c = Foo()
    util.Finalize(c, conn.send, args=('c',))
    d10 = Foo()
    util.Finalize(d10, conn.send, args=('d10',), exitpriority=1)
    d01 = Foo()
    util.Finalize(d01, conn.send, args=('d01',), exitpriority=0)
    d02 = Foo()
    util.Finalize(d02, conn.send, args=('d02',), exitpriority=0)
    d03 = Foo()
    util.Finalize(d03, conn.send, args=('d03',), exitpriority=0)
    util.Finalize(None, conn.send, args=('e',), exitpriority=-10)
    util.Finalize(None, conn.send, args=('STOP',), exitpriority=-100)
    util._exit_function()
    conn.close()
    os._exit(0)