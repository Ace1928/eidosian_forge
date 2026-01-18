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
def test_ssl_error_message(self):
    error = uuid.uuid4().hex
    self.stub_url('GET', exc=requests.exceptions.SSLError(error))
    session = client_session.Session()
    msg = 'SSL exception connecting to %(url)s: %(error)s' % {'url': self.TEST_URL, 'error': error}
    self.assertRaisesRegex(exceptions.SSLError, msg, session.get, self.TEST_URL)