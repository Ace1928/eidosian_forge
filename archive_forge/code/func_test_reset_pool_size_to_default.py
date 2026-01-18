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
def test_reset_pool_size_to_default(self):
    server = wsgi.Server(self.conf, 'test_resize', None, host='127.0.0.1', max_url_len=16384)
    server.start()
    server.stop()
    self.assertEqual(0, server._pool.size)
    server.reset()
    server.start()
    self.assertEqual(CONF.wsgi_default_pool_size, server._pool.size)