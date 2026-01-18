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
def test_oauth2_client_credential_method_other_not_rm_auth(self):
    base_https = self.TEST_URL.replace('http:', 'https:')
    other_auth_token = 'HW9bB6oYWJywz6mAN_KyIBXlof15Pk'
    self.stub_auth(json=self.TEST_APP_CRED_TOKEN_RESPONSE)
    with unittest.mock.patch('keystoneauth1.identity.v3.Password.get_headers') as co_mock:
        co_mock.return_value = {'X-Auth-Token': self.TEST_TOKEN, 'Authorization': other_auth_token}
        pass_auth = v3.Password(base_https, username=self.TEST_USER, password=self.TEST_PASS, include_catalog=False)
        sess = session.Session(auth=pass_auth)
        resp_ok = {'status_code': 200}
        self.requests_mock.post(f'{base_https}/test_api', [resp_ok])
        resp = sess.post(f'{base_https}/test_api', authenticated=True)
        self.assertRequestHeaderEqual('Authorization', other_auth_token)
        self.assertRequestHeaderEqual('X-Auth-Token', self.TEST_TOKEN)
        self.assertEqual(200, resp.status_code)