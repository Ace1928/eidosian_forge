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
class _TestPollEintr(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    @classmethod
    def _killer(cls, pid):
        time.sleep(0.1)
        os.kill(pid, signal.SIGUSR1)

    @unittest.skipUnless(hasattr(signal, 'SIGUSR1'), 'requires SIGUSR1')
    def test_poll_eintr(self):
        got_signal = [False]

        def record(*args):
            got_signal[0] = True
        pid = os.getpid()
        oldhandler = signal.signal(signal.SIGUSR1, record)
        try:
            killer = self.Process(target=self._killer, args=(pid,))
            killer.start()
            try:
                p = self.Process(target=time.sleep, args=(2,))
                p.start()
                p.join()
            finally:
                killer.join()
            self.assertTrue(got_signal[0])
            self.assertEqual(p.exitcode, 0)
        finally:
            signal.signal(signal.SIGUSR1, oldhandler)