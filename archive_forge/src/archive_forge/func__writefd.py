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
def _writefd(cls, conn, data, create_dummy_fds=False):
    if create_dummy_fds:
        for i in range(0, 256):
            if not cls._is_fd_assigned(i):
                os.dup2(conn.fileno(), i)
    fd = reduction.recv_handle(conn)
    if msvcrt:
        fd = msvcrt.open_osfhandle(fd, os.O_WRONLY)
    os.write(fd, data)
    os.close(fd)