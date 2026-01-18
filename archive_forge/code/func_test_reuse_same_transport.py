import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
def test_reuse_same_transport(self):
    possible_transports = []
    t1 = transport.get_transport_from_url('http://foo/', possible_transports=possible_transports)
    self.assertEqual([t1], possible_transports)
    t2 = transport.get_transport_from_url('http://foo/', possible_transports=[t1])
    self.assertIs(t1, t2)
    t3 = transport.get_transport_from_url('http://foo/path/')
    t4 = transport.get_transport_from_url('http://foo/path', possible_transports=[t3])
    self.assertIs(t3, t4)
    t5 = transport.get_transport_from_url('http://foo/path')
    t6 = transport.get_transport_from_url('http://foo/path/', possible_transports=[t5])
    self.assertIs(t5, t6)