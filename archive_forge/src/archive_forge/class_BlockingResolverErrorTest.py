import errno
import os
import signal
import socket
from subprocess import Popen
import sys
import time
import unittest
from tornado.netutil import (
from tornado.testing import AsyncTestCase, gen_test, bind_unused_port
from tornado.test.util import skipIfNoNetwork
import typing
class BlockingResolverErrorTest(AsyncTestCase, _ResolverErrorTestMixin):

    def setUp(self):
        super().setUp()
        self.resolver = BlockingResolver()
        self.real_getaddrinfo = socket.getaddrinfo
        socket.getaddrinfo = _failing_getaddrinfo

    def tearDown(self):
        socket.getaddrinfo = self.real_getaddrinfo
        super().tearDown()