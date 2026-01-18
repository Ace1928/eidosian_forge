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
def test_password_change_auth_state(self):
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    expired = ksa_utils.before_utcnow(days=2)
    token = fixture.V3Token(expires=expired)
    token_id = uuid.uuid4().hex
    state = json.dumps({'auth_token': token_id, 'body': token})
    a = v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS, user_domain_id=self.TEST_DOMAIN_ID, project_id=uuid.uuid4().hex)
    initial_cache_id = a.get_cache_id()
    self.assertIsNone(a.get_auth_state())
    a.set_auth_state(state)
    self.assertEqual(token_id, a.auth_ref.auth_token)
    s = session.Session()
    self.assertEqual(self.TEST_TOKEN, a.get_token(s))
    self.assertEqual(initial_cache_id, a.get_cache_id())