import copy
import json
import time
import unittest
import uuid
from keystoneauth1 import _utils as ksa_utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.exceptions import ClientException
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import base as v3_base
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_sends_nocatalog(self):
    del self.TEST_RESPONSE_DICT['token']['catalog']
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    a = v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS, include_catalog=False)
    s = session.Session(auth=a)
    s.get_token()
    auth_url = self.TEST_URL + '/auth/tokens'
    self.assertEqual(auth_url, a.token_url)
    self.assertEqual(auth_url + '?nocatalog', self.requests_mock.last_request.url)