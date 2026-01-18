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
def test_uri_length_limit(self):
    eventlet.monkey_patch(os=False, thread=False)
    server = wsgi.Server(self.conf, 'test_uri_length_limit', None, host='127.0.0.1', max_url_len=16384, port=33337)
    server.start()
    self.assertFalse(server._server.dead)
    uri = 'http://127.0.0.1:%d/%s' % (server.port, 10000 * 'x')
    resp = requests.get(uri, proxies={'http': ''})
    eventlet.sleep(0)
    self.assertNotEqual(requests.codes.REQUEST_URI_TOO_LARGE, resp.status_code)
    uri = 'http://127.0.0.1:%d/%s' % (server.port, 20000 * 'x')
    resp = requests.get(uri, proxies={'http': ''})
    eventlet.sleep(0)
    self.assertEqual(requests.codes.REQUEST_URI_TOO_LARGE, resp.status_code)
    server.stop()
    server.wait()