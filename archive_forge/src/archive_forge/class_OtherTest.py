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
@hashlib_helper.requires_hashdigest('md5')
class OtherTest(unittest.TestCase):

    def test_deliver_challenge_auth_failure(self):

        class _FakeConnection(object):

            def recv_bytes(self, size):
                return b'something bogus'

            def send_bytes(self, data):
                pass
        self.assertRaises(multiprocessing.AuthenticationError, multiprocessing.connection.deliver_challenge, _FakeConnection(), b'abc')

    def test_answer_challenge_auth_failure(self):

        class _FakeConnection(object):

            def __init__(self):
                self.count = 0

            def recv_bytes(self, size):
                self.count += 1
                if self.count == 1:
                    return multiprocessing.connection.CHALLENGE
                elif self.count == 2:
                    return b'something bogus'
                return b''

            def send_bytes(self, data):
                pass
        self.assertRaises(multiprocessing.AuthenticationError, multiprocessing.connection.answer_challenge, _FakeConnection(), b'abc')