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
def test_oauth2_mtls_client_credential_method_client_exception(self):
    base_https = self.TEST_URL.replace('http:', 'https:')
    oauth2_endpoint = f'{base_https}/OS-OAUTH2/token'
    a = v3.OAuth2mTlsClientCredential(self.TEST_URL, oauth2_endpoint=oauth2_endpoint, oauth2_client_id=self.TEST_CLIENT_CRED_ID)
    oauth2_post_resp = {'status_code': 400, 'json': {}}
    self.requests_mock.post(oauth2_endpoint, [oauth2_post_resp])
    sess = session.Session(auth=a)
    self.assertRaises(exceptions.ClientException, a.get_auth_ref, sess)