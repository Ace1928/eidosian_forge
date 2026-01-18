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
def test_oauth2_mtls_client_credential_method_resp_invalid_json(self):
    base_https = self.TEST_URL.replace('http:', 'https:')
    token_endpoint = f'{self.TEST_URL}/auth/tokens'
    oauth2_endpoint = f'{base_https}/OS-OAUTH2/token'
    oauth2_token = 'HW9bB6oYWJywz6mAN_KyIBXlof15Pk'
    a = v3.OAuth2mTlsClientCredential(self.TEST_URL, oauth2_endpoint=oauth2_endpoint, oauth2_client_id=self.TEST_CLIENT_CRED_ID)
    oauth2_post_resp = {'status_code': 200, 'json': {'access_token': oauth2_token, 'expires_in': 3600, 'token_type': 'Bearer'}}
    self.requests_mock.post(oauth2_endpoint, [oauth2_post_resp])
    token_verify_resp = {'status_code': 200, 'text': 'invalid json'}
    self.requests_mock.get(token_endpoint, [token_verify_resp])
    sess = session.Session(auth=a)
    self.assertRaises(exceptions.InvalidResponse, a.get_auth_ref, sess)