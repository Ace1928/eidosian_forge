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
@gen_test
def test_resolve_multiaddr(self):
    result = (yield self.resolver.resolve('google.com', 80, socket.AF_INET))
    self.assertIn((socket.AF_INET, ('1.2.3.4', 80)), result)
    result = (yield self.resolver.resolve('google.com', 80, socket.AF_INET6))
    self.assertIn((socket.AF_INET6, ('2a02:6b8:7c:40c:c51e:495f:e23a:3', 80, 0, 0)), result)