import os
import platform
import socket
import tempfile
import testtools
from unittest import mock
import eventlet
import eventlet.wsgi
import requests
import webob
from oslo_config import cfg
from oslo_service import sslutils
from oslo_service.tests import base
from oslo_service import wsgi
from oslo_utils import netutils
@testtools.skipIf(platform.mac_ver()[0] != '', 'SO_REUSEADDR behaves differently on OSX, see bug 1436895')
def test_socket_options_for_ssl_server(self):
    self.config(tcp_keepidle=500)
    server = wsgi.Server(self.conf, 'test_socket_options', None, host='127.0.0.1', port=0, use_ssl=True)
    server.start()
    sock = server.socket
    self.assertEqual(1, sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR))
    self.assertEqual(1, sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE))
    if hasattr(socket, 'TCP_KEEPIDLE'):
        self.assertEqual(CONF.tcp_keepidle, sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE))
    server.stop()
    server.wait()