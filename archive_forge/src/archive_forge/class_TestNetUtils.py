import collections
import contextlib
import errno
import os
import socket
import stat
import subprocess
import unittest
import psutil
import psutil.tests
from psutil import FREEBSD
from psutil import NETBSD
from psutil import POSIX
from psutil._common import open_binary
from psutil._common import open_text
from psutil._common import supports_ipv6
from psutil.tests import CI_TESTING
from psutil.tests import COVERAGE
from psutil.tests import HAS_CONNECTIONS_UNIX
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import PsutilTestCase
from psutil.tests import TestMemoryLeak
from psutil.tests import bind_socket
from psutil.tests import bind_unix_socket
from psutil.tests import call_until
from psutil.tests import chdir
from psutil.tests import create_sockets
from psutil.tests import filter_proc_connections
from psutil.tests import get_free_port
from psutil.tests import is_namedtuple
from psutil.tests import mock
from psutil.tests import process_namespace
from psutil.tests import reap_children
from psutil.tests import retry
from psutil.tests import retry_on_failure
from psutil.tests import safe_mkdir
from psutil.tests import safe_rmpath
from psutil.tests import serialrun
from psutil.tests import system_namespace
from psutil.tests import tcp_socketpair
from psutil.tests import terminate
from psutil.tests import unix_socketpair
from psutil.tests import wait_for_file
from psutil.tests import wait_for_pid
class TestNetUtils(PsutilTestCase):

    def bind_socket(self):
        port = get_free_port()
        with contextlib.closing(bind_socket(addr=('', port))) as s:
            self.assertEqual(s.getsockname()[1], port)

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_bind_unix_socket(self):
        name = self.get_testfn()
        sock = bind_unix_socket(name)
        with contextlib.closing(sock):
            self.assertEqual(sock.family, socket.AF_UNIX)
            self.assertEqual(sock.type, socket.SOCK_STREAM)
            self.assertEqual(sock.getsockname(), name)
            assert os.path.exists(name)
            assert stat.S_ISSOCK(os.stat(name).st_mode)
        name = self.get_testfn()
        sock = bind_unix_socket(name, type=socket.SOCK_DGRAM)
        with contextlib.closing(sock):
            self.assertEqual(sock.type, socket.SOCK_DGRAM)

    def tcp_tcp_socketpair(self):
        addr = ('127.0.0.1', get_free_port())
        server, client = tcp_socketpair(socket.AF_INET, addr=addr)
        with contextlib.closing(server):
            with contextlib.closing(client):
                self.assertEqual(server.getsockname(), addr)
                self.assertEqual(client.getpeername(), addr)
                self.assertNotEqual(client.getsockname(), addr)

    @unittest.skipIf(not POSIX, 'POSIX only')
    @unittest.skipIf(NETBSD or FREEBSD, '/var/run/log UNIX socket opened by default')
    def test_unix_socketpair(self):
        p = psutil.Process()
        num_fds = p.num_fds()
        self.assertEqual(filter_proc_connections(p.connections(kind='unix')), [])
        name = self.get_testfn()
        server, client = unix_socketpair(name)
        try:
            assert os.path.exists(name)
            assert stat.S_ISSOCK(os.stat(name).st_mode)
            self.assertEqual(p.num_fds() - num_fds, 2)
            self.assertEqual(len(filter_proc_connections(p.connections(kind='unix'))), 2)
            self.assertEqual(server.getsockname(), name)
            self.assertEqual(client.getpeername(), name)
        finally:
            client.close()
            server.close()

    def test_create_sockets(self):
        with create_sockets() as socks:
            fams = collections.defaultdict(int)
            types = collections.defaultdict(int)
            for s in socks:
                fams[s.family] += 1
                types[s.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)] += 1
            self.assertGreaterEqual(fams[socket.AF_INET], 2)
            if supports_ipv6():
                self.assertGreaterEqual(fams[socket.AF_INET6], 2)
            if POSIX and HAS_CONNECTIONS_UNIX:
                self.assertGreaterEqual(fams[socket.AF_UNIX], 2)
            self.assertGreaterEqual(types[socket.SOCK_STREAM], 2)
            self.assertGreaterEqual(types[socket.SOCK_DGRAM], 2)