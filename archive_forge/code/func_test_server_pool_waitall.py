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
def test_server_pool_waitall(self):
    server = wsgi.Server(self.conf, 'test_server', None, host='127.0.0.1')
    server.start()
    with mock.patch.object(server._pool, 'waitall') as mock_waitall:
        server.stop()
        server.wait()
        mock_waitall.assert_called_once_with()