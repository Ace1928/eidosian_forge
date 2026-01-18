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
def test_relpath_config_not_found(self):
    self.config(api_paste_config='api-paste.ini')
    self.assertRaises(wsgi.ConfigNotFound, wsgi.Loader, self.conf)