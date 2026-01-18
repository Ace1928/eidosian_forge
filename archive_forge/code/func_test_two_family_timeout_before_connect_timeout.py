from contextlib import closing
import getpass
import os
import socket
import unittest
from tornado.concurrent import Future
from tornado.netutil import bind_sockets, Resolver
from tornado.queues import Queue
from tornado.tcpclient import TCPClient, _Connector
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, gen_test
from tornado.test.util import skipIfNoIPv6, refusing_port, skipIfNonUnix
from tornado.gen import TimeoutError
import typing
def test_two_family_timeout_before_connect_timeout(self):
    conn, future = self.start_connect(self.addrinfo)
    self.assert_pending((AF1, 'a'))
    conn.on_timeout()
    self.assert_pending((AF1, 'a'), (AF2, 'c'))
    conn.on_connect_timeout()
    self.connect_futures.pop((AF1, 'a'))
    self.assertTrue(self.streams.pop('a').closed)
    self.connect_futures.pop((AF2, 'c'))
    self.assertTrue(self.streams.pop('c').closed)
    self.assert_pending()
    self.assertEqual(len(conn.streams), 2)
    self.assert_connector_streams_closed(conn)
    self.assertRaises(TimeoutError, future.result)