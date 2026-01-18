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
def test_ssl_server(self):

    def test_app(env, start_response):
        start_response('200 OK', {})
        return ['PONG']
    fake_ssl_server = wsgi.Server(self.conf, 'fake_ssl', test_app, host=self.host, port=0, use_ssl=True)
    fake_ssl_server.start()
    self.assertNotEqual(0, fake_ssl_server.port)
    response = requesting(method='GET', host=self.host, port=fake_ssl_server.port, ca_certs=os.path.join(SSL_CERT_DIR, 'ca.crt'))
    self.assertEqual('PONG', response[-4:])
    fake_ssl_server.stop()
    fake_ssl_server.wait()