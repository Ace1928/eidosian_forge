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
class TestConnectedSocket(ConnectionTestCase):
    """Test socket pairs which are actually connected to
    each other.
    """

    @unittest.skipIf(SUNOS, 'unreliable on SUONS')
    def test_tcp(self):
        addr = ('127.0.0.1', 0)
        self.assertEqual(this_proc_connections(kind='tcp4'), [])
        server, client = tcp_socketpair(AF_INET, addr=addr)
        try:
            cons = this_proc_connections(kind='tcp4')
            self.assertEqual(len(cons), 2)
            self.assertEqual(cons[0].status, psutil.CONN_ESTABLISHED)
            self.assertEqual(cons[1].status, psutil.CONN_ESTABLISHED)
        finally:
            server.close()
            client.close()

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_unix(self):
        testfn = self.get_testfn()
        server, client = unix_socketpair(testfn)
        try:
            cons = this_proc_connections(kind='unix')
            assert not (cons[0].laddr and cons[0].raddr), cons
            assert not (cons[1].laddr and cons[1].raddr), cons
            if NETBSD or FREEBSD:
                cons = [c for c in cons if c.raddr != '/var/run/log']
            self.assertEqual(len(cons), 2, msg=cons)
            if LINUX or FREEBSD or SUNOS or OPENBSD:
                self.assertEqual(cons[0].raddr, '')
                self.assertEqual(cons[1].raddr, '')
                self.assertEqual(testfn, cons[0].laddr or cons[1].laddr)
            else:
                self.assertEqual(cons[0].laddr or cons[1].laddr, testfn)
        finally:
            server.close()
            client.close()