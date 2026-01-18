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
def test_start_random_port(self):
    server = wsgi.Server(self.conf, 'test_random_port', None, host='127.0.0.1', port=0)
    server.start()
    self.assertNotEqual(0, server.port)
    server.stop()
    server.wait()