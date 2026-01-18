import datetime
import uuid
from oslo_utils import timeutils
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_oauth_access(self):
    consumer_id = uuid.uuid4().hex
    access_token_id = uuid.uuid4().hex
    token = fixture.V3Token()
    token.set_project_scope()
    token.set_oauth(access_token_id=access_token_id, consumer_id=consumer_id)
    auth_ref = access.create(body=token)
    self.assertEqual(consumer_id, auth_ref.oauth_consumer_id)
    self.assertEqual(access_token_id, auth_ref.oauth_access_token_id)
    self.assertEqual(consumer_id, auth_ref._data['token']['OS-OAUTH1']['consumer_id'])
    self.assertEqual(access_token_id, auth_ref._data['token']['OS-OAUTH1']['access_token_id'])