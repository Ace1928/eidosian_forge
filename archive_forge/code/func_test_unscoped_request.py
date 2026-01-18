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
def test_unscoped_request(self):
    token = fixture.V3Token()
    self.stub_auth(json=token)
    password = uuid.uuid4().hex
    a = v3.Password(self.TEST_URL, user_id=token.user_id, password=password, unscoped=True)
    s = session.Session()
    auth_ref = a.get_access(s)
    self.assertFalse(auth_ref.scoped)
    body = self.requests_mock.last_request.json()
    ident = body['auth']['identity']
    self.assertEqual(['password'], ident['methods'])
    self.assertEqual(token.user_id, ident['password']['user']['id'])
    self.assertEqual(password, ident['password']['user']['password'])
    self.assertEqual('unscoped', body['auth']['scope'])