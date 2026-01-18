from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_submit_acl_with_users_project_access_stripped_uuid(self):
    bad_href = 'http://badsite.com/secrets/' + self.secret_uuid
    self.test_should_submit_acl_with_users_project_access_set(bad_href)