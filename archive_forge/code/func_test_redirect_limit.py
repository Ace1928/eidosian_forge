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
def test_redirect_limit(self):
    self.setup_redirects()
    for i in (1, 2):
        session = client_session.Session(redirect=i)
        resp = session.get(self.REDIRECT_CHAIN[0])
        self.assertEqual(resp.status_code, 305)
        self.assertEqual(resp.url, self.REDIRECT_CHAIN[i])
        self.assertEqual(resp.text, self.DEFAULT_REDIRECT_BODY)