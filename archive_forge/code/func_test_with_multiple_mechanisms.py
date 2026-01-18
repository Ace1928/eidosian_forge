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
def test_with_multiple_mechanisms(self):
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    p = v3.PasswordMethod(username=self.TEST_USER, password=self.TEST_PASS)
    t = v3.TokenMethod(token='foo')
    a = v3.Auth(self.TEST_URL, [p, t], trust_id='trust')
    self.assertTrue(a.has_scope_parameters)
    s = session.Session(auth=a)
    self.assertEqual({'X-Auth-Token': self.TEST_TOKEN}, s.get_auth_headers())
    req = {'auth': {'identity': {'methods': ['password', 'token'], 'password': {'user': {'name': self.TEST_USER, 'password': self.TEST_PASS}}, 'token': {'id': 'foo'}}, 'scope': {'OS-TRUST:trust': {'id': 'trust'}}}}
    self.assertRequestBodyIs(json=req)
    self.assertEqual(s.auth.auth_ref.auth_token, self.TEST_TOKEN)