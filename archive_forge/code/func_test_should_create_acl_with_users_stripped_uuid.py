from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_create_acl_with_users_stripped_uuid(self):
    bad_href = 'http://badsite.com/containers/' + self.container_uuid
    self.test_should_create_acl_with_users(bad_href)