import os
import socket
import textwrap
import unittest
from contextlib import closing
from socket import AF_INET
from socket import AF_INET6
from socket import SOCK_DGRAM
from socket import SOCK_STREAM
import psutil
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import supports_ipv6
from psutil._compat import PY3
from psutil.tests import AF_UNIX
from psutil.tests import HAS_CONNECTIONS_UNIX
from psutil.tests import SKIP_SYSCONS
from psutil.tests import PsutilTestCase
from psutil.tests import bind_socket
from psutil.tests import bind_unix_socket
from psutil.tests import check_connection_ntuple
from psutil.tests import create_sockets
from psutil.tests import filter_proc_connections
from psutil.tests import reap_children
from psutil.tests import retry_on_failure
from psutil.tests import serialrun
from psutil.tests import skip_on_access_denied
from psutil.tests import tcp_socketpair
from psutil.tests import unix_socketpair
from psutil.tests import wait_for_file
@serialrun
class TestUnconnectedSockets(ConnectionTestCase):
    """Tests sockets which are open but not connected to anything."""

    def get_conn_from_sock(self, sock):
        cons = this_proc_connections(kind='all')
        smap = dict([(c.fd, c) for c in cons])
        if NETBSD or FREEBSD:
            return smap[sock.fileno()]
        else:
            self.assertEqual(len(cons), 1)
            if cons[0].fd != -1:
                self.assertEqual(smap[sock.fileno()].fd, sock.fileno())
            return cons[0]

    def check_socket(self, sock):
        """Given a socket, makes sure it matches the one obtained
        via psutil. It assumes this process created one connection
        only (the one supposed to be checked).
        """
        conn = self.get_conn_from_sock(sock)
        check_connection_ntuple(conn)
        if conn.fd != -1:
            self.assertEqual(conn.fd, sock.fileno())
        self.assertEqual(conn.family, sock.family)
        self.assertEqual(conn.type, sock.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE))
        laddr = sock.getsockname()
        if not laddr and PY3 and isinstance(laddr, bytes):
            laddr = laddr.decode()
        if sock.family == AF_INET6:
            laddr = laddr[:2]
        self.assertEqual(conn.laddr, laddr)
        if sock.family == AF_UNIX and HAS_CONNECTIONS_UNIX:
            cons = this_proc_connections(kind='all')
            self.compare_procsys_connections(os.getpid(), cons, kind='all')
        return conn

    def test_tcp_v4(self):
        addr = ('127.0.0.1', 0)
        with closing(bind_socket(AF_INET, SOCK_STREAM, addr=addr)) as sock:
            conn = self.check_socket(sock)
            self.assertEqual(conn.raddr, ())
            self.assertEqual(conn.status, psutil.CONN_LISTEN)

    @unittest.skipIf(not supports_ipv6(), 'IPv6 not supported')
    def test_tcp_v6(self):
        addr = ('::1', 0)
        with closing(bind_socket(AF_INET6, SOCK_STREAM, addr=addr)) as sock:
            conn = self.check_socket(sock)
            self.assertEqual(conn.raddr, ())
            self.assertEqual(conn.status, psutil.CONN_LISTEN)

    def test_udp_v4(self):
        addr = ('127.0.0.1', 0)
        with closing(bind_socket(AF_INET, SOCK_DGRAM, addr=addr)) as sock:
            conn = self.check_socket(sock)
            self.assertEqual(conn.raddr, ())
            self.assertEqual(conn.status, psutil.CONN_NONE)

    @unittest.skipIf(not supports_ipv6(), 'IPv6 not supported')
    def test_udp_v6(self):
        addr = ('::1', 0)
        with closing(bind_socket(AF_INET6, SOCK_DGRAM, addr=addr)) as sock:
            conn = self.check_socket(sock)
            self.assertEqual(conn.raddr, ())
            self.assertEqual(conn.status, psutil.CONN_NONE)

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_unix_tcp(self):
        testfn = self.get_testfn()
        with closing(bind_unix_socket(testfn, type=SOCK_STREAM)) as sock:
            conn = self.check_socket(sock)
            self.assertEqual(conn.raddr, '')
            self.assertEqual(conn.status, psutil.CONN_NONE)

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_unix_udp(self):
        testfn = self.get_testfn()
        with closing(bind_unix_socket(testfn, type=SOCK_STREAM)) as sock:
            conn = self.check_socket(sock)
            self.assertEqual(conn.raddr, '')
            self.assertEqual(conn.status, psutil.CONN_NONE)