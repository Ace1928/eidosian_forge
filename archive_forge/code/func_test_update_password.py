from unittest import mock
import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import users
def test_update_password(self):
    old_password = uuid.uuid4().hex
    new_password = uuid.uuid4().hex
    self.stub_url('POST', [self.collection_key, self.TEST_USER_ID, 'password'])
    self.client.user_id = self.TEST_USER_ID
    self.manager.update_password(old_password, new_password)
    exp_req_body = {'user': {'password': new_password, 'original_password': old_password}}
    self.assertEqual('%s/users/%s/password' % (self.TEST_URL, self.TEST_USER_ID), self.requests_mock.last_request.url)
    self.assertRequestBodyIs(json=exp_req_body)
    self.assertNotIn(old_password, self.logger.output)
    self.assertNotIn(new_password, self.logger.output)