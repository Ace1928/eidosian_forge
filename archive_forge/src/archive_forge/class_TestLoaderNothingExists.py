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
class TestLoaderNothingExists(WsgiTestCase):
    """Loader tests where os.path.exists always returns False."""

    def setUp(self):
        super(TestLoaderNothingExists, self).setUp()
        mock_patcher = mock.patch.object(os.path, 'exists', lambda _: False)
        mock_patcher.start()
        self.addCleanup(mock_patcher.stop)

    def test_relpath_config_not_found(self):
        self.config(api_paste_config='api-paste.ini')
        self.assertRaises(wsgi.ConfigNotFound, wsgi.Loader, self.conf)

    def test_asbpath_config_not_found(self):
        self.config(api_paste_config='/etc/openstack-srv/api-paste.ini')
        self.assertRaises(wsgi.ConfigNotFound, wsgi.Loader, self.conf)