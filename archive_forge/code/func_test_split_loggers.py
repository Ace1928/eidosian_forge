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
def test_split_loggers(self):

    def get_logger_io(name):
        logger_name = 'keystoneauth.session.{name}'.format(name=name)
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        string_io = io.StringIO()
        handler = logging.StreamHandler(string_io)
        logger.addHandler(handler)
        return string_io
    io_dict = {}
    for name in ('request', 'body', 'response', 'request-id'):
        io_dict[name] = get_logger_io(name)
    auth = AuthPlugin()
    sess = client_session.Session(auth=auth, split_loggers=True)
    response_key = uuid.uuid4().hex
    response_val = uuid.uuid4().hex
    response = {response_key: response_val}
    request_id = uuid.uuid4().hex
    self.stub_url('GET', json=response, headers={'Content-Type': 'application/json', 'X-OpenStack-Request-ID': request_id})
    resp = sess.get(self.TEST_URL, headers={encodeutils.safe_encode('x-bytes-header'): encodeutils.safe_encode('bytes-value')})
    self.assertEqual(response, resp.json())
    request_output = io_dict['request'].getvalue().strip()
    response_output = io_dict['response'].getvalue().strip()
    body_output = io_dict['body'].getvalue().strip()
    id_output = io_dict['request-id'].getvalue().strip()
    self.assertIn('curl -g -i -X GET {url}'.format(url=self.TEST_URL), request_output)
    self.assertIn('-H "x-bytes-header: bytes-value"', request_output)
    self.assertEqual('[200] Content-Type: application/json X-OpenStack-Request-ID: {id}'.format(id=request_id), response_output)
    self.assertEqual('GET call to {url} used request id {id}'.format(url=self.TEST_URL, id=request_id), id_output)
    self.assertEqual('{{"{key}": "{val}"}}'.format(key=response_key, val=response_val), body_output)