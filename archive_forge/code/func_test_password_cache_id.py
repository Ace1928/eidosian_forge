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
def test_password_cache_id(self):
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    project_name = uuid.uuid4().hex
    a = v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS, user_domain_id=self.TEST_DOMAIN_ID, project_domain_name=self.TEST_DOMAIN_NAME, project_name=project_name)
    b = v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS, user_domain_id=self.TEST_DOMAIN_ID, project_domain_name=self.TEST_DOMAIN_NAME, project_name=project_name)
    a_id = a.get_cache_id()
    b_id = b.get_cache_id()
    self.assertEqual(a_id, b_id)
    c = v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS, user_domain_id=self.TEST_DOMAIN_ID, project_domain_name=self.TEST_DOMAIN_NAME, project_id=project_name)
    c_id = c.get_cache_id()
    self.assertNotEqual(a_id, c_id)
    self.assertIsNone(a.get_auth_state())
    self.assertIsNone(b.get_auth_state())
    self.assertIsNone(c.get_auth_state())
    s = session.Session()
    self.assertEqual(self.TEST_TOKEN, a.get_token(s))
    self.assertTrue(self.requests_mock.called)