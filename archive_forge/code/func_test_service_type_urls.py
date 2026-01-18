import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_service_type_urls(self):
    service_type = 'compute'
    interface = 'public'
    path = '/instances'
    status = 200
    body = 'SUCCESS'
    self.stub_service_url(service_type=service_type, interface=interface, path=path, status_code=status, text=body)
    sess = client_session.Session(auth=AuthPlugin())
    resp = sess.get(path, endpoint_filter={'service_type': service_type, 'interface': interface})
    self.assertEqual(self.requests_mock.last_request.url, AuthPlugin.SERVICE_URLS['compute']['public'] + path)
    self.assertEqual(resp.text, body)
    self.assertEqual(resp.status_code, status)