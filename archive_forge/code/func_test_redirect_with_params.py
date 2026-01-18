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
def test_redirect_with_params(self):
    params = {'foo': 'bar'}
    session = client_session.Session(redirect=True)
    self.setup_redirects(final_kwargs={'complete_qs': True})
    resp = session.get(self.REDIRECT_CHAIN[0], params=params)
    self.assertResponse(resp)
    self.assertTrue(len(resp.history), len(self.REDIRECT_CHAIN))
    self.assertQueryStringIs(None)