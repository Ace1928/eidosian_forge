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
def test_oauth2_client_credential_method_https(self):
    self.TEST_URL = self.TEST_URL.replace('http:', 'https:')
    base_https = self.TEST_URL
    oauth2_endpoint = f'{base_https}/oauth_token'
    oauth2_token = 'HW9bB6oYWJywz6mAN_KyIBXlof15Pk'
    self.stub_auth(json=self.TEST_APP_CRED_TOKEN_RESPONSE)
    client_cre = v3.OAuth2ClientCredential(base_https, oauth2_endpoint=oauth2_endpoint, oauth2_client_id=self.TEST_CLIENT_CRED_ID, oauth2_client_secret=self.TEST_CLIENT_CRED_SECRET)
    oauth2_resp = {'status_code': 200, 'json': {'access_token': oauth2_token, 'expires_in': 3600, 'token_type': 'Bearer'}}
    self.requests_mock.post(oauth2_endpoint, [oauth2_resp])
    sess = session.Session(auth=client_cre)
    initial_cache_id = client_cre.get_cache_id()
    auth_head = sess.get_auth_headers()
    self.assertEqual(self.TEST_TOKEN, auth_head['X-Auth-Token'])
    self.assertEqual(f'Bearer {oauth2_token}', auth_head['Authorization'])
    self.assertEqual(sess.auth.auth_ref.auth_token, self.TEST_TOKEN)
    self.assertEqual(initial_cache_id, client_cre.get_cache_id())
    resp_ok = {'status_code': 200}
    self.requests_mock.post(f'{base_https}/test_api', [resp_ok])
    resp = sess.post(f'{base_https}/test_api', authenticated=True)
    self.assertRequestHeaderEqual('Authorization', f'Bearer {oauth2_token}')
    self.assertRequestHeaderEqual('X-Auth-Token', self.TEST_TOKEN)
    self.assertEqual(200, resp.status_code)