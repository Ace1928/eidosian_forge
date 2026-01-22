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
def _test_resource_tracker(self):
    cmd = 'if 1:\n            import time, os, tempfile\n            import multiprocess as mp\n            from multiprocess import resource_tracker\n            from multiprocess.shared_memory import SharedMemory\n\n            mp.set_start_method("spawn")\n            rand = tempfile._RandomNameSequence()\n\n\n            def create_and_register_resource(rtype):\n                if rtype == "semaphore":\n                    lock = mp.Lock()\n                    return lock, lock._semlock.name\n                elif rtype == "shared_memory":\n                    sm = SharedMemory(create=True, size=10)\n                    return sm, sm._name\n                else:\n                    raise ValueError(\n                        "Resource type {{}} not understood".format(rtype))\n\n\n            resource1, rname1 = create_and_register_resource("{rtype}")\n            resource2, rname2 = create_and_register_resource("{rtype}")\n\n            os.write({w}, rname1.encode("ascii") + b"\\n")\n            os.write({w}, rname2.encode("ascii") + b"\\n")\n\n            time.sleep(10)\n        '
    for rtype in resource_tracker._CLEANUP_FUNCS:
        with self.subTest(rtype=rtype):
            if rtype == 'noop':
                continue
            r, w = os.pipe()
            p = subprocess.Popen([sys.executable, '-E', '-c', cmd.format(w=w, rtype=rtype)], pass_fds=[w], stderr=subprocess.PIPE)
            os.close(w)
            with open(r, 'rb', closefd=True) as f:
                name1 = f.readline().rstrip().decode('ascii')
                name2 = f.readline().rstrip().decode('ascii')
            _resource_unlink(name1, rtype)
            p.terminate()
            p.wait()
            deadline = time.monotonic() + support.LONG_TIMEOUT
            while time.monotonic() < deadline:
                time.sleep(0.5)
                try:
                    _resource_unlink(name2, rtype)
                except OSError as e:
                    self.assertIn(e.errno, (errno.ENOENT, errno.EINVAL))
                    break
            else:
                raise AssertionError(f'A {rtype} resource was leaked after a process was abruptly terminated.')
            err = p.stderr.read().decode('utf-8')
            p.stderr.close()
            expected = 'resource_tracker: There appear to be 2 leaked {} objects'.format(rtype)
            self.assertRegex(err, expected)
            self.assertRegex(err, 'resource_tracker: %r: \\[Errno' % name1)