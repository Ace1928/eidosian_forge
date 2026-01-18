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
def test_oauth2_client_credential_reauth_called_https(self):
    base_https = self.TEST_URL.replace('http:', 'https:')
    oauth2_endpoint = f'{base_https}/oauth_token'
    oauth2_token = 'HW9bB6oYWJywz6mAN_KyIBXlof15Pk'
    auth = v3.OAuth2ClientCredential(base_https, oauth2_endpoint=oauth2_endpoint, oauth2_client_id='clientcredid', oauth2_client_secret='secret')
    oauth2_resp = {'status_code': 200, 'json': {'access_token': oauth2_token, 'expires_in': 3600, 'token_type': 'Bearer'}}
    self.requests_mock.post(oauth2_endpoint, [oauth2_resp])
    sess = session.Session(auth=auth)
    resp_text = json.dumps(self.TEST_APP_CRED_TOKEN_RESPONSE)
    resp_ok = {'status_code': 200, 'headers': {'Content-Type': 'application/json', 'x-subject-token': self.TEST_TOKEN}, 'text': resp_text}
    self.requests_mock.post(f'{base_https}/auth/tokens', [resp_ok, {'text': 'Failed', 'status_code': 401}, resp_ok])
    resp = sess.post(f'{base_https}/auth/tokens', authenticated=True)
    self.assertRequestHeaderEqual('Authorization', f'Bearer {oauth2_token}')
    self.assertEqual(200, resp.status_code)
    self.assertEqual(resp_text, resp.text)