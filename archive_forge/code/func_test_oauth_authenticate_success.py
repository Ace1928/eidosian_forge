import uuid
from oauthlib import oauth1
from testtools import matchers
from keystoneauth1.extras import oauth1 as ksa_oauth1
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils as test_utils
def test_oauth_authenticate_success(self):
    consumer_key = uuid.uuid4().hex
    consumer_secret = uuid.uuid4().hex
    access_key = uuid.uuid4().hex
    access_secret = uuid.uuid4().hex
    oauth_token = fixture.V3Token(methods=['oauth1'], oauth_consumer_id=consumer_key, oauth_access_token_id=access_key)
    oauth_token.set_project_scope()
    self.stub_auth(json=oauth_token)
    a = ksa_oauth1.V3OAuth1(self.TEST_URL, consumer_key=consumer_key, consumer_secret=consumer_secret, access_key=access_key, access_secret=access_secret)
    s = session.Session(auth=a)
    t = s.get_token()
    self.assertEqual(self.TEST_TOKEN, t)
    OAUTH_REQUEST_BODY = {'auth': {'identity': {'methods': ['oauth1'], 'oauth1': {}}}}
    self.assertRequestBodyIs(json=OAUTH_REQUEST_BODY)
    req_headers = self.requests_mock.last_request.headers
    oauth_client = oauth1.Client(consumer_key, client_secret=consumer_secret, resource_owner_key=access_key, resource_owner_secret=access_secret, signature_method=oauth1.SIGNATURE_HMAC)
    self._validate_oauth_headers(req_headers['Authorization'], oauth_client)