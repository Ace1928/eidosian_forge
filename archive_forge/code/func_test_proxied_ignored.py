import threading
import time
from unittest import mock
from oslo_config import fixture as config
from oslo_serialization import jsonutils
from oslotest import base as test_base
import requests
import webob.dec
import webob.exc
from oslo_middleware import healthcheck
from oslo_middleware.healthcheck import __main__
def test_proxied_ignored(self):
    conf = {'ignore_proxied_requests': True}
    modern_headers = {'x-forwarded': 'https://localhost'}
    self._do_test(conf, expected_code=webob.exc.HTTPOk.code, expected_body=b'Hello, World!!!', headers=modern_headers)
    legacy_headers = {'x-forwarded-proto': 'https', 'x-forwarded-host': 'localhost', 'x-forwarded-for': '192.0.2.11'}
    self._do_test(conf, expected_code=webob.exc.HTTPOk.code, expected_body=b'Hello, World!!!', headers=legacy_headers)