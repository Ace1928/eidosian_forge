import datetime
import gettext
import http.client as http
import os
import socket
from unittest import mock
import eventlet.patcher
import fixtures
from oslo_concurrency import processutils
from oslo_serialization import jsonutils
import routes
import webob
from glance.api.v2 import router as router_v2
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance import i18n
from glance.image_cache import prefetcher
from glance.tests import utils as test_utils
@mock.patch.object(prefetcher, 'Prefetcher')
@mock.patch.object(wsgi.Server, 'configure_socket')
def test_reserved_stores_not_allowed(self, mock_configure_socket, mock_prefetcher):
    """Ensure the reserved stores are not allowed"""
    enabled_backends = {'os_glance_file_store': 'file'}
    self.config(enabled_backends=enabled_backends)
    server = wsgi.Server(threads=1, initialize_glance_store=True)
    self.assertRaises(RuntimeError, server.configure)